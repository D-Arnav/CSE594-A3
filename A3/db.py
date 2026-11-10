"""Persistence: Turso when TURSO_DATABASE_URL is set, otherwise a local SQLite file."""
import json
import os
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import libsql

import env  # noqa: F401  (loads .env)

LOCAL_DB = Path(__file__).parent / "study.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    participant_id TEXT PRIMARY KEY, ai_assist INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attention_checks (
    participant_id TEXT NOT NULL, check_index INTEGER NOT NULL, answer TEXT NOT NULL, passed INTEGER NOT NULL,
    created_at TEXT NOT NULL, PRIMARY KEY (participant_id, check_index));
CREATE TABLE IF NOT EXISTS trials (
    participant_id TEXT NOT NULL, trial_index INTEGER NOT NULL, image_id TEXT NOT NULL, ai_assist INTEGER NOT NULL,
    response TEXT NOT NULL, correct INTEGER NOT NULL, chose_rank INTEGER, expanded_other INTEGER NOT NULL,
    ai_source TEXT, ai_top1 TEXT, ai_top1_correct INTEGER, rt_ms INTEGER NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (participant_id, trial_index));
CREATE TABLE IF NOT EXISTS surveys (
    participant_id TEXT PRIMARY KEY, confidence INTEGER NOT NULL, mental_demand INTEGER NOT NULL,
    geo_knowledge INTEGER NOT NULL, seen_other_condition INTEGER NOT NULL, trust INTEGER, usefulness INTEGER,
    free_text TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ai_shown (
    participant_id TEXT NOT NULL, trial_index INTEGER NOT NULL, image_id TEXT NOT NULL, source TEXT NOT NULL,
    suggestions TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (participant_id, trial_index));
"""


def connect():
    url = os.environ.get("TURSO_DATABASE_URL")
    if url:
        return libsql.connect(url, auth_token=os.environ["TURSO_AUTH_TOKEN"])
    return libsql.connect(str(LOCAL_DB))


def now():
    return datetime.now(timezone.utc).isoformat()


def _write(sql, params):
    with closing(connect()) as conn:
        conn.execute(sql, params)
        conn.commit()


def _one(sql, params):
    with closing(connect()) as conn:
        return conn.execute(sql, params).fetchone()


def init():
    with closing(connect()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def add_participant(participant_id, ai_assist):
    # IGNORE: a double click on start creates one session, not an error
    _write("INSERT OR IGNORE INTO participants VALUES (?, ?, ?)", (participant_id, int(ai_assist), now()))


def get_participant(participant_id):
    """-> {"ai_assist", "done", "survey_done", "checks_done", "check_failures"}, or None for an unknown id."""
    row = _one("""SELECT ai_assist,
                         (SELECT COUNT(*) FROM trials t WHERE t.participant_id = p.participant_id),
                         EXISTS (SELECT 1 FROM surveys s WHERE s.participant_id = p.participant_id),
                         (SELECT COUNT(*) FROM attention_checks c WHERE c.participant_id = p.participant_id),
                         (SELECT COUNT(*) FROM attention_checks c WHERE c.participant_id = p.participant_id AND NOT c.passed)
                  FROM participants p WHERE participant_id = ?""", (participant_id,))
    return row and {"ai_assist": bool(row[0]), "done": row[1], "survey_done": bool(row[2]),
                    "checks_done": row[3], "check_failures": row[4]}


def add_attention_check(participant_id, check_index, answer, passed):
    _write("INSERT OR IGNORE INTO attention_checks VALUES (?, ?, ?, ?, ?)",
           (participant_id, check_index, answer, int(passed), now()))


def add_trial(participant_id, trial_index, image_id, ai_assist, response, correct, chose_rank, expanded_other,
              ai_source, ai_top1, ai_top1_correct, rt_ms):
    # the primary key plus IGNORE makes a double submit harmless: the first answer stands
    _write("INSERT OR IGNORE INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
           (participant_id, trial_index, image_id, int(ai_assist), response, int(correct), chose_rank,
            int(expanded_other), ai_source, ai_top1, ai_top1_correct, rt_ms, now()))


def add_survey(participant_id, confidence, mental_demand, geo_knowledge, seen_other_condition, trust, usefulness,
               free_text):
    _write("INSERT OR IGNORE INTO surveys VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
           (participant_id, confidence, mental_demand, geo_knowledge, int(seen_other_condition), trust, usefulness,
            free_text, now()))


def get_ai_shown(participant_id, trial_index):
    """-> {"source", "suggestions"} shown on this trial, or None if the AI hasn't run for it yet."""
    row = _one("SELECT source, suggestions FROM ai_shown WHERE participant_id = ? AND trial_index = ?",
               (participant_id, trial_index))
    return row and {"source": row[0], "suggestions": json.loads(row[1])}


def save_ai_shown(participant_id, trial_index, image_id, source, suggestions):
    """Log the AI output for a trial and return what is logged. The first one stands, so a refresh shows the same."""
    _write("INSERT OR IGNORE INTO ai_shown VALUES (?, ?, ?, ?, ?, ?)",
           (participant_id, trial_index, image_id, source, json.dumps(suggestions), now()))
    return get_ai_shown(participant_id, trial_index)
