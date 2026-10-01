"""GeoGuess-HAI study interface (SPEC 11). No GPU: reads the cached study_dataset.csv.

Run from repo root:  streamlit run study_app/app.py
URL params: ?pid=N fixes the participant id (else the next free id in the DB; on Streamlit Cloud
            always pass ?pid=N, the DB is wiped on restart);
            ?live=1 asks server.py (env GEOHAI_SERVER_URL) for AI outputs, falling back to the cache.
Env overrides: GEOHAI_STUDY_CSV, GEOHAI_PRACTICE_CSV, GEOHAI_DB.
Optional Google Sheets logging: st.secrets [gsheets] sheet_url = "...", plus a
[gsheets.service_account] table; the sheet needs worksheets
named "responses" and "surveys". SQLite is always written.

rt_ms is measured server-side, from the first script run that renders the trial (after the
AI outputs are fetched) to the run that handles Submit, so it includes network/render latency.
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from assignments import BLOCK_SIZE, HUMAN, HUMAN_AI, build_sequence, choose_practice, make_sets  # noqa: E402
from common import CLUSTERS, OUTPUTS, ROOT, countries, load_config  # noqa: E402

STUDY_CSV = Path(os.environ.get("GEOHAI_STUDY_CSV", OUTPUTS / "study_dataset.csv"))
PRACTICE_CSV = Path(os.environ.get("GEOHAI_PRACTICE_CSV", OUTPUTS / "dev" / "dev_dataset.csv"))
DB_PATH = Path(os.environ.get("GEOHAI_DB", HERE / "responses.db"))
LIVE_TIMEOUT_S = 20

BANNER = "The AI can be wrong and sometimes describes things that aren't in the image. Check its evidence."
COLS = {
    "responses": ["participant_id", "session_id", "block", "condition", "trial_order", "trial_id",
                  "response", "correct", "chose_rank", "ai_top1_correct", "rt_ms", "expanded_other",
                  "timestamp", "is_practice", "ai_source"],
    "surveys": ["participant_id", "session_id", "trust", "usefulness", "geo_knowledge",
                "free_text", "timestamp"],
}


# ---------- logging ----------

def connect(db_path):
    conn = sqlite3.connect(db_path)
    for table, cols in COLS.items():
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols)})")
    conn.execute("CREATE TABLE IF NOT EXISTS participants "
                 "(session_id TEXT PRIMARY KEY, participant_id INTEGER, timestamp TEXT)")
    return conn


def log_row(db_path, table, row):
    cols = COLS[table]
    with closing(connect(db_path)) as conn, conn:
        conn.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})",
                     [row.get(c) for c in cols])


def read_table(db_path, table):
    with closing(connect(db_path)) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def register_participant(db_path, session_id, pid=None):
    """Record the session; pid None -> next id after the largest one used so far (atomic)."""
    with closing(connect(db_path)) as conn, conn:
        if pid is None:
            conn.execute("INSERT INTO participants SELECT ?, COALESCE(MAX(participant_id), -1) + 1, ? "
                         "FROM participants", (session_id, now()))
        else:
            conn.execute("INSERT INTO participants VALUES (?, ?, ?)", (session_id, pid, now()))
        return conn.execute("SELECT participant_id FROM participants WHERE session_id = ?",
                            (session_id,)).fetchone()[0]


def log_sheet(table, row):
    """Best-effort mirror to Google Sheets; never breaks the study."""
    try:
        conf = st.secrets["gsheets"]
    except Exception:
        return
    try:
        import gspread
        gc = gspread.service_account_from_dict(dict(conf["service_account"]))
        ws = gc.open_by_url(conf["sheet_url"]).worksheet(table)
        ws.append_row(["" if row.get(c) is None else row.get(c) for c in COLS[table]])
    except Exception as e:  # ponytail: errors go to server logs only, participants never see them
        print(f"gsheets logging failed: {e!r}", file=sys.stderr)


def log(table, row):
    log_row(DB_PATH, table, row)
    log_sheet(table, row)


def now():
    return datetime.now(timezone.utc).isoformat()


# ---------- trial data ----------

def cached_ai(row):
    return [{"rank": i, "country": row[f"ai_top{i}"], "p": float(row[f"ai_p{i}"]),
             "rationale": "" if pd.isna(row[f"ai_rationale{i}"]) else str(row[f"ai_rationale{i}"])}
            for i in (1, 2, 3)]


def fetch_live_ai(trial_id, url, timeout=LIVE_TIMEOUT_S):
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=json.dumps({"trial_id": trial_id}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        top3 = json.load(r)["top3"]
    return [{"rank": i + 1, "country": t["country"], "p": float(t["p"]), "rationale": t.get("rationale") or ""}
            for i, t in enumerate(top3[:3])]


def trial_view(row, condition, live_url=None):
    """What the participant sees. Human-only views carry no AI fields (and never the ground truth)."""
    view = {"trial_id": row["trial_id"], "image_path": str(ROOT / row["image_path"]), "condition": condition}
    if condition == HUMAN_AI:
        view["ai"], view["ai_source"] = cached_ai(row), "cache"
        if live_url:
            try:
                view["ai"], view["ai_source"] = fetch_live_ai(row["trial_id"], live_url), "live"
            except Exception as e:
                print(f"live AI failed for {row['trial_id']}, using cache: {e!r}", file=sys.stderr)
    return view


def response_row(trial, view, row, response, chose_rank, expanded_other, rt_ms, pid, session_id):
    ai_top1 = view["ai"][0]["country"] if "ai" in view else row["ai_top1"]
    return {"participant_id": pid, "session_id": session_id, "block": trial["block"],
            "condition": trial["condition"], "trial_order": trial["trial_order"], "trial_id": trial["trial_id"],
            "response": response, "correct": int(response == row["country"]), "chose_rank": chose_rank,
            "ai_top1_correct": int(ai_top1 == row["country"]), "rt_ms": rt_ms,
            "expanded_other": int(expanded_other), "timestamp": now(),
            "is_practice": int(trial["is_practice"]), "ai_source": view.get("ai_source")}


@st.cache_data
def load_study(study_csv, practice_csv, block_size, seed):
    study = pd.read_csv(study_csv)
    study["ai_correct_top1"] = study["ai_correct_top1"].astype(str).str.lower().isin(["true", "1"])
    sets = make_sets(study.to_dict("records"), block_size, seed)
    rows = {r["trial_id"]: r for r in study.to_dict("records")}
    practice = None
    if Path(practice_csv).exists():
        dev = pd.read_csv(practice_csv)
        rows.update({r["trial_id"]: r for r in dev.to_dict("records")})
        practice = choose_practice(dev["trial_id"], seed)
    return rows, sets, practice


# ---------- UI ----------

INSTRUCTIONS = """
In this short study (about 10 minutes) you will see street-level photos and guess **which country**
each was taken in, choosing from 20 countries.

- There are two parts of {n} photos each. In one part you decide on your own. In the other, an AI
  suggests three countries with its confidence and the evidence it sees. You make the final choice.
- Take about 20–40 seconds per photo. There is no feedback on whether you were right.
- First you get one practice photo for each part. Practice photos are not scored.
- Afterwards there is a 1-minute survey.

Participation is voluntary and anonymous: we record only your answers, response times and survey
ratings. You may stop at any time by closing the tab.
"""
CONDITION_TEXT = {
    HUMAN: "**On your own:** pick the country you think this photo was taken in, then press Submit.",
    HUMAN_AI: "**With AI suggestions:** pick one of the AI's three suggestions, or choose another country, "
              "then press Submit.",
}


def select(country, rank):
    st.session_state.cur.update(choice=country, rank=rank)


def country_grid(cur, active, key, rank):
    for cluster, members in CLUSTERS.items():
        names = [c for c in members if c in active]
        if not names:
            continue
        st.caption("Other" if cluster == "Backup" else cluster)
        cols = st.columns(2)
        for i, c in enumerate(names):
            picked = cur["choice"] == c and cur["rank"] == rank
            cols[i % 2].button(c, key=f"{key}_g_{c}", on_click=select, args=(c, rank),
                               type="primary" if picked else "secondary", width="stretch")


def ai_cards(cur, view, key):
    st.info(BANNER)
    for card in view["ai"]:
        with st.container(border=True):
            st.markdown(f"**{card['rank']}. {card['country']}**")
            st.progress(min(max(card["p"], 0.0), 1.0), text=f"{card['p']:.0%} confidence")
            st.write(card["rationale"])
            picked = cur["rank"] == card["rank"]
            st.button(f"Choose {card['country']}", key=f"{key}_c_{card['rank']}", on_click=select,
                      args=(card["country"], card["rank"]), type="primary" if picked else "secondary")


def trial_screen(rows, active, live_url):
    ss = st.session_state
    trial = ss.seq[ss.i]
    row = rows[trial["trial_id"]]
    key = f"t{ss.i}"
    if ss.get("cur", {}).get("key") != key:  # first render of this trial
        with st.spinner("Loading..."):
            view = trial_view(row, trial["condition"], live_url)
        ss.cur = {"key": key, "view": view, "t0": time.monotonic(), "choice": None, "rank": None,
                  "expanded": False}
    cur, view = ss.cur, ss.cur["view"]

    in_block = [t for t in ss.seq if t["block"] == trial["block"]]
    if trial["is_practice"]:
        st.subheader("Practice photo (not scored)")
    else:
        st.subheader(f"Part {trial['block']} — photo {in_block.index(trial) + 1} of {len(in_block)}")
    st.markdown(CONDITION_TEXT[trial["condition"]])

    left, right = st.columns([3, 2])
    left.image(view["image_path"], width="stretch")
    left.caption("Tip: hover over the photo and use the fullscreen button to zoom in.")
    with right:
        if trial["condition"] == HUMAN:
            country_grid(cur, active, key, None)
        else:
            ai_cards(cur, view, key)
            if st.button("None of these — choose another", key=f"{key}_other", type="tertiary"):
                cur["expanded"] = True
            if cur["expanded"]:
                country_grid(cur, active, key, 0)

    st.markdown(f"Your answer: **{cur['choice']}**" if cur["choice"] else "Your answer: *none selected*")
    if st.button("Submit", key=f"{key}_submit", type="primary", disabled=cur["choice"] is None):
        rt_ms = int((time.monotonic() - cur["t0"]) * 1000)
        log("responses", response_row(trial, view, row, cur["choice"], cur["rank"], cur["expanded"], rt_ms,
                                      ss.pid, ss.session_id))
        ss.i += 1
        if ss.i == len(ss.seq):
            ss.stage = "survey"
        st.rerun()


def survey_screen():
    ss = st.session_state
    st.subheader("Almost done: a few questions")
    scale = list(range(1, 8))
    with st.form("survey"):
        trust = st.radio("How much did you trust the AI's suggestions? (1 = not at all, 7 = completely)",
                         scale, index=None, horizontal=True)
        useful = st.radio("How useful were the AI's suggestions? (1 = not at all, 7 = very)",
                          scale, index=None, horizontal=True)
        geo = st.radio("How would you rate your own geography knowledge? (1 = very low, 7 = expert)",
                       scale, index=None, horizontal=True)
        text = st.text_area("Anything else you'd like to tell us? (optional)")
        if st.form_submit_button("Finish", type="primary"):
            if None in (trust, useful, geo):
                st.error("Please answer the three rating questions.")
            else:
                log("surveys", {"participant_id": ss.pid, "session_id": ss.session_id, "trust": trust,
                                "usefulness": useful, "geo_knowledge": geo, "free_text": text,
                                "timestamp": now()})
                ss.stage = "done"
                st.rerun()


def consent_screen(sets, practice, seed):
    ss = st.session_state
    st.title("Where was this photo taken?")
    st.markdown(INSTRUCTIONS.format(n=len(sets[0])))
    agree = st.checkbox("I am 18 or older and agree to take part.")
    if st.button("Start", type="primary", disabled=not agree):
        ss.session_id = uuid.uuid4().hex
        qpid = st.query_params.get("pid", "")
        ss.pid = register_participant(DB_PATH, ss.session_id, int(qpid) if qpid.isdigit() else None)
        ss.seq = build_sequence(ss.pid, sets, practice, seed)
        ss.i = 0
        ss.stage = "trials"
        st.rerun()


def main():
    st.set_page_config(page_title="GeoGuess study", layout="wide")
    if not STUDY_CSV.exists():
        st.error(f"Missing {STUDY_CSV}. Run predict.py first.")
        st.stop()
    cfg = load_config()
    seed = cfg["seed"]
    block_size = cfg.get("study", {}).get("block_size", BLOCK_SIZE)
    rows, sets, practice = load_study(str(STUDY_CSV), str(PRACTICE_CSV), block_size, seed)
    live_url = os.environ.get("GEOHAI_SERVER_URL") if st.query_params.get("live") == "1" else None

    stage = st.session_state.setdefault("stage", "consent")
    if stage == "consent":
        consent_screen(sets, practice, seed)
    elif stage == "trials":
        trial_screen(rows, countries(cfg), live_url)
    elif stage == "survey":
        survey_screen()
    else:
        st.success("Thank you for taking part! You can close this tab now.")
        st.caption(f"Completion code: {st.session_state.session_id[:8]}")


if __name__ == "__main__":
    main()
