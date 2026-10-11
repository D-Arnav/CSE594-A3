"""Study server: one HTML page and a small JSON API.   uvicorn app:app --port 8501

The browser never gets the true country, human-only trials carry no AI fields, and correctness is computed here.
"""
import random
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
SEED, BLOCK = 42, 8
HUMAN, HUMAN_AI = "human_only", "human_ai"
COUNTRIES = ["United States", "Germany", "Japan", "South Korea", "India"]
RESPONSE_COLS = ["participant_id", "session_id", "block", "condition", "trial_order", "trial_id", "response",
                 "correct", "chose_rank", "ai_top1_correct", "rt_ms", "expanded_other", "timestamp", "is_practice"]
SURVEY_COLS = ["participant_id", "session_id", "trust", "usefulness", "geo_knowledge", "free_text", "timestamp"]

study = pd.read_csv(ROOT / "data/study_dataset.csv", dtype={"trial_id": str}, keep_default_na=False)
dev = pd.read_csv(ROOT / "data/dev_dataset.csv", dtype={"trial_id": str}, keep_default_na=False)
ROWS = {r["trial_id"]: r for r in pd.concat([study, dev]).to_dict("records")}

# 25 sets of 8: sort by (AI right?, country, seeded tiebreak) and deal round-robin, so every set gets a similar mix
rng = random.Random(SEED)
order = [k[-1] for k in sorted((str(r["ai_correct_top1"]) == "True", r["country"], rng.random(), r["trial_id"])
                               for r in study.to_dict("records"))]
N_SETS = len(order) // BLOCK
SETS = [order[j:N_SETS * BLOCK:N_SETS] for j in range(N_SETS)]
rng.shuffle(SETS)  # set numbers carry no information about AI accuracy
PRACTICE = dict(zip((HUMAN, HUMAN_AI), random.Random(SEED).sample(sorted(dev.trial_id), 2)))  # not study photos


def sequence(p):
    """Practice x2, then Block A and Block B for participant p. Over every cycle of 50 participants each set
    appears once per condition and block position; participants 2k and 2k+1 see the same two sets with the
    conditions swapped, so every photo is seen in both conditions and nobody sees a photo twice."""
    cycle, slot = divmod(p, 2 * N_SETS)
    first, second = slot // 2, (slot // 2 + 1 + cycle % (N_SETS - 1)) % N_SETS
    blocks = [(first, HUMAN), (second, HUMAN_AI)] if p % 2 == 0 else [(first, HUMAN_AI), (second, HUMAN)]
    shuffle = random.Random(SEED * 100003 + p).shuffle
    seq = [{"block": "practice", "condition": c, "trial_id": PRACTICE[c], "is_practice": True} for _, c in blocks]
    for name, (s, c) in zip("AB", blocks):
        ids = list(SETS[s])
        shuffle(ids)
        seq += [{"block": name, "condition": c, "trial_id": t, "is_practice": False} for t in ids]
    return [dict(t, trial_order=i) for i, t in enumerate(seq)]


def view(t):
    """What the browser gets for one trial."""
    v = {**{k: t[k] for k in ("trial_order", "block", "condition", "is_practice", "trial_id")},
         "image": f"/img/{t['trial_id']}"}
    if t["condition"] == HUMAN_AI:
        r = ROWS[t["trial_id"]]
        v["ai"] = [{"rank": i, "country": r[f"ai_top{i}"], "p": float(r[f"ai_p{i}"]),
                    "rationale": str(r[f"ai_rationale{i}"])} for i in (1, 2, 3)]
    return v


def db():
    conn = sqlite3.connect(ROOT / "responses.db")
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS participants (participant_id INTEGER PRIMARY KEY, session_id TEXT, timestamp TEXT);
        CREATE TABLE IF NOT EXISTS responses ({", ".join(RESPONSE_COLS)});
        CREATE TABLE IF NOT EXISTS surveys ({", ".join(SURVEY_COLS)});
        CREATE UNIQUE INDEX IF NOT EXISTS one_response ON responses (participant_id, trial_order);
        CREATE UNIQUE INDEX IF NOT EXISTS one_survey ON surveys (participant_id);""")
    return conn


def insert(conn, table, cols, row):
    try:  # the unique indexes turn double submits (two tabs, double clicks) into a 409
        conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", [row[c] for c in cols])
    except sqlite3.IntegrityError:
        raise HTTPException(409, "already logged")


def now():
    return datetime.now(timezone.utc).isoformat()


def state(conn, pid):
    found = conn.execute("SELECT session_id FROM participants WHERE participant_id = ?", (pid,)).fetchone()
    if not found:
        raise HTTPException(404, "unknown participant")
    return {"pid": pid, "session_id": found[0], "countries": COUNTRIES,
            "done": conn.execute("SELECT COUNT(*) FROM responses WHERE participant_id = ?", (pid,)).fetchone()[0],
            "survey_done": bool(conn.execute("SELECT 1 FROM surveys WHERE participant_id = ?", (pid,)).fetchone()),
            "trials": [view(t) for t in sequence(pid)]}


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
Pid = Annotated[int, Field(ge=0, lt=2**63)]  # SQLite integer range


class NewSession(BaseModel):
    pid: int | None = Field(None, ge=0, le=100_000)


class Response(BaseModel):
    pid: Pid
    trial_order: int
    trial_id: str
    response: str
    chose_rank: int | None = None
    expanded_other: bool = False
    rt_ms: int = Field(ge=0, le=3_600_000)


class Survey(BaseModel):
    pid: Pid
    trust: int = Field(ge=1, le=7)
    usefulness: int = Field(ge=1, le=7)
    geo_knowledge: int = Field(ge=1, le=7)
    free_text: str = Field("", max_length=5000)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/img/{trial_id}")
def image(trial_id: str):
    if trial_id not in ROWS:
        raise HTTPException(404)
    return FileResponse(ROOT / ROWS[trial_id]["image_path"])


@app.get("/api/session")
def get_session(pid: Pid):
    with closing(db()) as conn:
        return state(conn, pid)


@app.post("/api/session")
def new_session(req: NewSession):
    """Register a participant: the next free id (one atomic statement) unless pid is given; a known pid resumes."""
    with closing(db()) as conn, conn:
        if req.pid is None:
            cur = conn.execute("INSERT INTO participants SELECT COALESCE(MAX(participant_id), -1) + 1, ?, ? "
                               "FROM participants", (uuid.uuid4().hex, now()))
            pid = conn.execute("SELECT participant_id FROM participants WHERE rowid = ?", (cur.lastrowid,)).fetchone()[0]
        else:
            pid = req.pid
            conn.execute("INSERT OR IGNORE INTO participants VALUES (?, ?, ?)", (pid, uuid.uuid4().hex, now()))
    with closing(db()) as conn:
        return state(conn, pid)


@app.post("/api/response")
def log_response(req: Response):
    with closing(db()) as conn, conn:
        st = state(conn, req.pid)
        if req.trial_order != st["done"] or st["done"] >= len(st["trials"]):
            raise HTTPException(409, f"expected trial {st['done']}")
        trial = st["trials"][req.trial_order]
        if req.trial_id != trial["trial_id"]:
            raise HTTPException(422, "trial_id does not match this participant's sequence")
        row = ROWS[req.trial_id]
        if req.response not in COUNTRIES:
            raise HTTPException(422, "response not in label space")
        if trial["condition"] == HUMAN_AI:
            if req.chose_rank not in (0, 1, 2, 3) or (req.chose_rank == 0 and not req.expanded_other):
                raise HTTPException(422, "chose_rank must be 1-3, or 0 after expanding 'None of these'")
            if req.chose_rank and req.response != row[f"ai_top{req.chose_rank}"]:
                raise HTTPException(422, "response does not match the chosen AI card")
        elif req.chose_rank is not None or req.expanded_other:
            raise HTTPException(422, "human-only trials have no AI choice")
        insert(conn, "responses", RESPONSE_COLS, {
            "participant_id": req.pid, "session_id": st["session_id"], "block": trial["block"],
            "condition": trial["condition"], "trial_order": req.trial_order, "trial_id": req.trial_id,
            "response": req.response, "correct": int(req.response == row["country"]), "chose_rank": req.chose_rank,
            "ai_top1_correct": int(row["ai_top1"] == row["country"]), "rt_ms": req.rt_ms,
            "expanded_other": int(req.expanded_other), "timestamp": now(), "is_practice": int(trial["is_practice"])})
    return {"done": req.trial_order + 1}


@app.post("/api/survey")
def log_survey(req: Survey):
    with closing(db()) as conn, conn:
        st = state(conn, req.pid)
        if st["done"] < len(st["trials"]):
            raise HTTPException(409, "trials not finished")
        if st["survey_done"]:
            raise HTTPException(409, "survey already submitted")
        insert(conn, "surveys", SURVEY_COLS, {**req.model_dump(), "participant_id": req.pid, "session_id": st["session_id"],
                                 "timestamp": now()})
    return {"ok": True}
