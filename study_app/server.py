"""GeoGuess-HAI study interface (SPEC 11): a static page plus a tiny JSON API. No GPU.

Run from repo root:  uvicorn study_app.server:app --port 8501
Open http://localhost:8501/?pid=N to fix the participant id (else the next free id is taken at consent).
Env overrides: GEOHAI_STUDY_CSV, GEOHAI_PRACTICE_CSV, GEOHAI_DB.

The browser never receives the true country. Human-only trials carry no AI fields at all.
Correctness is computed here when a response is logged.
"""
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from common import OUTPUTS, ROOT, countries, load_config  # noqa: E402

from study_app.assignments import BLOCK_SIZE, HUMAN_AI, build_sequence, choose_practice, make_sets  # noqa: E402

RESPONSE_COLS = ["participant_id", "session_id", "block", "condition", "trial_order", "trial_id", "response",
                 "correct", "chose_rank", "ai_top1_correct", "rt_ms", "expanded_other", "timestamp", "is_practice"]
SURVEY_COLS = ["participant_id", "session_id", "trust", "usefulness", "geo_knowledge", "free_text", "timestamp"]


def paths():
    return (os.environ.get("GEOHAI_STUDY_CSV", str(OUTPUTS / "study_dataset.csv")),
            os.environ.get("GEOHAI_PRACTICE_CSV", str(OUTPUTS / "dev" / "dev_dataset.csv")))


@lru_cache  # ponytail: keyed on the CSV paths; restart the server after regenerating the data
def load(study_csv, practice_csv):
    cfg = load_config()
    seed = cfg["seed"]
    study = pd.read_csv(study_csv)
    study["ai_correct_top1"] = study["ai_correct_top1"].astype(str).str.lower().isin(["true", "1"])
    sets = make_sets(study.to_dict("records"), cfg.get("study", {}).get("block_size", BLOCK_SIZE), seed)
    rows = {r["trial_id"]: r for r in study.to_dict("records")}
    practice = None
    if Path(practice_csv).exists():
        dev = pd.read_csv(practice_csv)
        rows.update({r["trial_id"]: r for r in dev.to_dict("records")})
        practice = choose_practice(dev["trial_id"], seed)
    return {"rows": rows, "sets": sets, "practice": practice, "seed": seed, "countries": countries(cfg)}


def data():
    return load(*paths())


def db():
    conn = sqlite3.connect(os.environ.get("GEOHAI_DB", str(HERE / "responses.db")))
    conn.execute("CREATE TABLE IF NOT EXISTS participants "
                 "(participant_id INTEGER PRIMARY KEY, session_id TEXT, timestamp TEXT)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS responses ({', '.join(RESPONSE_COLS)})")
    conn.execute(f"CREATE TABLE IF NOT EXISTS surveys ({', '.join(SURVEY_COLS)})")
    # double submits (two tabs, double clicks) fail here instead of logging twice
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_response ON responses (participant_id, trial_order)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_survey ON surveys (participant_id)")
    return conn


def insert(conn, table, cols, row):
    try:
        conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [row[c] for c in cols])
    except sqlite3.IntegrityError:
        raise HTTPException(409, "already logged")


def now():
    return datetime.now(timezone.utc).isoformat()


def sequence(pid):
    d = data()
    return build_sequence(pid, d["sets"], d["practice"], d["seed"])


def trial_view(trial, row):
    """What the browser gets for one trial: never the true country; AI fields only in the human-AI condition."""
    view = {k: trial[k] for k in ("trial_order", "block", "condition", "is_practice", "trial_id")}
    view["image"] = f"/img/{trial['trial_id']}"
    if trial["condition"] == HUMAN_AI:
        view["ai"] = [{"rank": i, "country": row[f"ai_top{i}"], "p": float(row[f"ai_p{i}"]),
                       "rationale": "" if pd.isna(row[f"ai_rationale{i}"]) else str(row[f"ai_rationale{i}"])}
                      for i in (1, 2, 3)]
    return view


def state(conn, pid):
    found = conn.execute("SELECT session_id FROM participants WHERE participant_id = ?", (pid,)).fetchone()
    if not found:
        raise HTTPException(404, "unknown participant")
    rows = data()["rows"]
    return {"pid": pid, "session_id": found[0], "countries": data()["countries"],
            "done": conn.execute("SELECT COUNT(*) FROM responses WHERE participant_id = ?", (pid,)).fetchone()[0],
            "survey_done": bool(conn.execute("SELECT 1 FROM surveys WHERE participant_id = ?", (pid,)).fetchone()),
            "trials": [trial_view(t, rows[t["trial_id"]]) for t in sequence(pid)]}


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class NewSession(BaseModel):
    pid: int | None = Field(None, ge=0, le=100_000)


class Response(BaseModel):
    pid: int
    trial_order: int
    trial_id: str
    response: str
    chose_rank: int | None = None
    expanded_other: bool = False
    rt_ms: int = Field(ge=0, le=3_600_000)


class Survey(BaseModel):
    pid: int
    trust: int = Field(ge=1, le=7)
    usefulness: int = Field(ge=1, le=7)
    geo_knowledge: int = Field(ge=1, le=7)
    free_text: str = Field("", max_length=5000)


@app.get("/api/session")
def get_session(pid: int):
    with closing(db()) as conn:
        return state(conn, pid)


@app.post("/api/session")
def new_session(req: NewSession):
    """Register a participant (next free id if pid is None; atomic in one statement). Existing pid -> resume."""
    with closing(db()) as conn, conn:
        sid = uuid.uuid4().hex
        if req.pid is None:
            cur = conn.execute("INSERT INTO participants SELECT COALESCE(MAX(participant_id), -1) + 1, ?, ? "
                               "FROM participants", (sid, now()))
            pid = conn.execute("SELECT participant_id FROM participants WHERE rowid = ?",
                               (cur.lastrowid,)).fetchone()[0]
        else:
            pid = req.pid
            conn.execute("INSERT OR IGNORE INTO participants VALUES (?, ?, ?)", (pid, sid, now()))
    with closing(db()) as conn:
        return state(conn, pid)


@app.post("/api/response")
def log_response(req: Response):
    d = data()
    with closing(db()) as conn, conn:
        st = state(conn, req.pid)
        seq = sequence(req.pid)
        if req.trial_order != st["done"]:
            raise HTTPException(409, f"expected trial {st['done']}")
        trial = seq[req.trial_order]
        if req.trial_id != trial["trial_id"]:
            raise HTTPException(422, "trial_id does not match this participant's sequence")
        if req.response not in d["countries"]:
            raise HTTPException(422, "response not in label space")
        row = d["rows"][trial["trial_id"]]
        if trial["condition"] == HUMAN_AI:
            if req.chose_rank not in (0, 1, 2, 3) or (req.chose_rank == 0 and not req.expanded_other):
                raise HTTPException(422, "chose_rank must be 1-3, or 0 after expanding 'None of these'")
            if req.chose_rank and req.response != row[f"ai_top{req.chose_rank}"]:
                raise HTTPException(422, "response does not match the chosen AI card")
        elif req.chose_rank is not None or req.expanded_other:
            raise HTTPException(422, "human-only trials have no AI choice")
        insert(conn, "responses", RESPONSE_COLS, {
            "participant_id": req.pid, "session_id": st["session_id"], "block": trial["block"],
            "condition": trial["condition"], "trial_order": trial["trial_order"], "trial_id": trial["trial_id"],
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
        insert(conn, "surveys", SURVEY_COLS, {**req.model_dump(), "participant_id": req.pid,
                                               "session_id": st["session_id"], "timestamp": now()})
    return {"ok": True}


@app.get("/img/{trial_id}")
def image(trial_id: str):
    row = data()["rows"].get(trial_id)
    if row is None:
        raise HTTPException(404)
    path = Path(row["image_path"])
    return FileResponse(path if path.is_absolute() else ROOT / path)


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
