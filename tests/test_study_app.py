import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "study_app"))
import app  # noqa: E402
from assignments import HUMAN, HUMAN_AI  # noqa: E402
from common import CLUSTER_OF, countries, load_config  # noqa: E402

ACTIVE = countries(load_config())


def fake_row(i, prefix, tmp_path):
    truth = ACTIVE[i % len(ACTIVE)]
    top = [ACTIVE[(i + k) % len(ACTIVE)] for k in ((1, 0, 2) if i % 2 else (0, 1, 2))]
    img = tmp_path / f"{prefix}{i:03d}.jpg"
    Image.new("RGB", (64, 48), (i * 10 % 255, 100, 150)).save(img)
    row = {"trial_id": f"{prefix}{i:03d}", "image_path": str(img), "source_id": str(i), "country": truth,
           "cluster": CLUSTER_OF[truth], "has_text": False, "ai_correct_top1": top[0] == truth}
    for k, c in enumerate(top, 1):
        row |= {f"ai_top{k}": c, f"ai_p{k}": [0.5, 0.3, 0.1][k - 1], f"ai_rationale{k}": f"RATIONALE-{i}-{k}"}
    return row


@pytest.fixture
def data(tmp_path, monkeypatch):
    study = pd.DataFrame([fake_row(i, "S", tmp_path) for i in range(1, 17)])
    dev = pd.DataFrame([fake_row(i, "D", tmp_path) for i in range(1, 4)])
    study.to_csv(tmp_path / "study.csv", index=False)
    dev.to_csv(tmp_path / "dev.csv", index=False)
    monkeypatch.setenv("GEOHAI_STUDY_CSV", str(tmp_path / "study.csv"))
    monkeypatch.setenv("GEOHAI_PRACTICE_CSV", str(tmp_path / "dev.csv"))
    monkeypatch.setenv("GEOHAI_DB", str(tmp_path / "r.db"))
    return tmp_path, study.set_index("trial_id", drop=False).to_dict("index")


def test_log_roundtrip(tmp_path):
    db = tmp_path / "r.db"
    row = {"participant_id": 3, "session_id": "abc", "block": "A", "condition": HUMAN_AI, "trial_order": 2,
           "trial_id": "S001", "response": "Chile", "correct": 1, "chose_rank": 0, "ai_top1_correct": 0,
           "rt_ms": 1234, "expanded_other": 1, "timestamp": app.now(), "is_practice": 0, "ai_source": "cache"}
    app.log_row(db, "responses", row)
    app.log_row(db, "surveys", {"participant_id": 3, "session_id": "abc", "trust": 5, "usefulness": 6,
                                "geo_knowledge": 2, "free_text": "hi", "timestamp": app.now()})
    got = app.read_table(db, "responses")
    assert got.to_dict("records") == [row]
    assert app.read_table(db, "surveys").loc[0, "trust"] == 5


def test_register_participant_autoincrements(tmp_path):
    db = tmp_path / "r.db"
    assert app.register_participant(db, "a") == 0
    assert app.register_participant(db, "b") == 1
    assert app.register_participant(db, "c", pid=7) == 7
    assert app.register_participant(db, "d") == 8


def test_human_only_view_has_no_ai_fields(data):
    _, rows = data
    row = rows["S003"]
    view = app.trial_view(row, HUMAN, live_url="http://127.0.0.1:9")
    assert set(view) == {"trial_id", "image_path", "condition"}
    text = json.dumps(view)
    assert "RATIONALE" not in text and row["country"] not in text
    ai = app.trial_view(row, HUMAN_AI)
    assert [c["rank"] for c in ai["ai"]] == [1, 2, 3] and ai["ai"][0]["country"] == row["ai_top1"]
    assert ai["ai_source"] == "cache"


def test_live_mode_and_fallback(data):
    _, rows = data

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = {"trial_id": body["trial_id"], "top3": [{"country": c, "p": 0.2, "rationale": "live"}
                                                          for c in ("Japan", "Chile", "Italy")]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        view = app.trial_view(rows["S001"], HUMAN_AI, f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
    assert view["ai_source"] == "live" and view["ai"][0] == {"rank": 1, "country": "Japan", "p": 0.2,
                                                              "rationale": "live"}
    view = app.trial_view(rows["S001"], HUMAN_AI, "http://127.0.0.1:9")  # nothing listening
    assert view["ai_source"] == "cache" and view["ai"][0]["country"] == rows["S001"]["ai_top1"]


def test_response_row(data):
    _, rows = data
    row = rows["S002"]  # even i -> AI top-1 correct
    trial = {"block": "B", "condition": HUMAN_AI, "trial_order": 9, "trial_id": "S002", "is_practice": False}
    view = app.trial_view(row, HUMAN_AI)
    r = app.response_row(trial, view, row, row["country"], 1, False, 500, 4, "s")
    assert (r["correct"], r["chose_rank"], r["ai_top1_correct"], r["is_practice"]) == (1, 1, 1, 0)
    assert list(r) == app.COLS["responses"]


def page_text(at):
    return " ".join(str(e.value) for kind in ("markdown", "info", "caption", "subheader", "title")
                    for e in getattr(at, kind))


def answer(at):
    """Pick the first available option on the current trial and submit it."""
    cards = [b for b in at.button if "_c_" in (b.key or "")]
    if cards:
        cards[0].click().run()
    else:
        next(b for b in at.button if "_g_" in (b.key or "")).click().run()
    next(b for b in at.button if (b.key or "").endswith("_submit")).click().run()


def test_app_full_flow(data):
    from streamlit.testing.v1 import AppTest
    tmp_path, rows = data
    at = AppTest.from_file(str(ROOT / "study_app" / "app.py"), default_timeout=30)
    at.query_params["pid"] = "0"
    at.run()
    assert not at.exception
    assert at.title[0].value == "Where was this photo taken?"
    at.checkbox[0].check().run()
    next(b for b in at.button if b.label == "Start").click().run()

    # pid 0 -> practice human-only first: no AI information anywhere on the page
    assert "Practice" in at.subheader[0].value
    text = page_text(at)
    assert app.BANNER not in text and "RATIONALE" not in text and "confidence" not in text
    assert not [b for b in at.button if "None of these" in b.label]
    answer(at)

    # practice human-AI: banner + three cards
    assert app.BANNER in page_text(at)
    assert len([b for b in at.button if "_c_" in (b.key or "")]) == 3
    # "None of these" expands the 20-country grid; a grid pick logs chose_rank 0
    assert not [b for b in at.button if "_g_" in (b.key or "")]
    next(b for b in at.button if "None of these" in b.label).click().run()
    grid = [b for b in at.button if "_g_" in (b.key or "")]
    assert sorted(b.label for b in grid) == sorted(ACTIVE)
    grid[0].click().run()
    next(b for b in at.button if (b.key or "").endswith("_submit")).click().run()
    for _ in range(16):
        assert not at.exception
        answer(at)

    assert "questions" in at.subheader[0].value
    for r in at.radio:
        r.set_value(4)
    at.text_area[0].input("fun")
    at.button[0].click().run()
    assert at.success

    resp = app.read_table(tmp_path / "r.db", "responses")
    assert len(resp) == 18 and resp.is_practice.sum() == 2
    assert list(resp.block) == ["practice"] * 2 + ["A"] * 8 + ["B"] * 8
    assert set(resp[resp.block == "A"].condition) == {HUMAN} and set(resp[resp.block == "B"].condition) == {HUMAN_AI}
    assert resp.participant_id.eq(0).all() and resp.rt_ms.ge(0).all()
    ai = resp[resp.condition == HUMAN_AI]
    assert list(ai.chose_rank) == [0] + [1] * 8 and list(ai.expanded_other) == [1] + [0] * 8
    with sqlite3.connect(tmp_path / "r.db") as conn:
        assert conn.execute("SELECT trust, free_text FROM surveys").fetchall() == [(4, "fun")]
