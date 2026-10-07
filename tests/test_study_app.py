import json
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from common import CLUSTER_OF, countries, load_config
from study_app import server
from study_app.assignments import HUMAN, HUMAN_AI

ACTIVE = countries(load_config())


def fake_row(i, prefix, tmp_path):
    truth = ACTIVE[i % len(ACTIVE)]
    top = [ACTIVE[(i + k) % len(ACTIVE)] for k in ((1, 0, 2) if i % 2 else (0, 1, 2))]
    img = tmp_path / f"{prefix}{i:03d}.jpg"
    Image.new("RGB", (64, 48), (i * 10 % 255, 100, 150)).save(img)
    row = {"trial_id": f"{prefix}{i:03d}", "image_path": str(img), "country": truth, "cluster": CLUSTER_OF[truth],
           "ai_correct_top1": top[0] == truth}
    for k, c in enumerate(top, 1):
        row |= {f"ai_top{k}": c, f"ai_p{k}": [0.5, 0.3, 0.1][k - 1], f"ai_rationale{k}": f"RATIONALE-{i}-{k}"}
    return row


@pytest.fixture
def env(tmp_path, monkeypatch):
    study = pd.DataFrame([fake_row(i, "S", tmp_path) for i in range(1, 17)])  # 2 sets of 8
    dev = pd.DataFrame([fake_row(i, "D", tmp_path) for i in range(1, 4)])
    study.to_csv(tmp_path / "study.csv", index=False)
    dev.to_csv(tmp_path / "dev.csv", index=False)
    monkeypatch.setenv("GEOHAI_STUDY_CSV", str(tmp_path / "study.csv"))
    monkeypatch.setenv("GEOHAI_PRACTICE_CSV", str(tmp_path / "dev.csv"))
    monkeypatch.setenv("GEOHAI_DB", str(tmp_path / "r.db"))
    rows = {r["trial_id"]: r for r in pd.concat([study, dev]).to_dict("records")}
    return TestClient(server.app), rows, tmp_path / "r.db"


def answer(t, rows, right=True):
    """A valid response body: human-AI picks card 1 (or the 'Other' row), human-only picks a country."""
    truth = rows[t["trial_id"]]["country"]
    resp = truth if right else next(c for c in ACTIVE if c != truth)
    body = {"trial_order": t["trial_order"], "trial_id": t["trial_id"], "response": resp, "rt_ms": 1500}
    if t["condition"] == HUMAN_AI:
        rank = next((a["rank"] for a in t["ai"] if a["country"] == resp), 0)
        body |= {"chose_rank": rank, "expanded_other": rank == 0}
    return body


def test_session_create_autoincrement_and_resume(env):
    c, rows, _ = env
    s0 = c.post("/api/session", json={}).json()
    s1 = c.post("/api/session", json={}).json()
    assert (s0["pid"], s1["pid"]) == (0, 1) and s0["session_id"] != s1["session_id"]
    assert c.post("/api/session", json={"pid": 7}).json()["pid"] == 7
    assert c.post("/api/session", json={}).json()["pid"] == 8
    again = c.post("/api/session", json={"pid": 0}).json()  # existing pid -> same session, no new row
    assert again["session_id"] == s0["session_id"]
    assert c.get("/api/session", params={"pid": 3}).status_code == 404
    seq = s0["trials"]
    assert [t["block"] for t in seq] == ["practice"] * 2 + ["A"] * 8 + ["B"] * 8
    assert [t["condition"] for t in seq[2:]] == [HUMAN] * 8 + [HUMAN_AI] * 8  # even pid: human-only first
    assert s1["trials"][2]["condition"] == HUMAN_AI
    assert s0["countries"] == ACTIVE

    c.post("/api/response", json={"pid": 0, **answer(seq[0], rows)})
    resumed = c.get("/api/session", params={"pid": 0}).json()
    assert resumed["done"] == 1 and not resumed["survey_done"]


def test_payload_never_leaks_truth_and_human_only_has_no_ai(env):
    c, rows, _ = env
    s = c.post("/api/session", json={}).json()
    for t in s["trials"]:
        row = rows[t["trial_id"]]
        assert "country" not in t and "correct" not in json.dumps(t)
        if t["condition"] == HUMAN:
            assert set(t) == {"trial_order", "block", "condition", "is_practice", "trial_id", "image"}
        else:
            assert [a["country"] for a in t["ai"]] == [row["ai_top1"], row["ai_top2"], row["ai_top3"]]
            assert t["ai"][0]["rationale"] == row["ai_rationale1"]
    human = [t for t in s["trials"] if t["condition"] == HUMAN]
    assert "RATIONALE" not in json.dumps(human)
    img = c.get(s["trials"][0]["image"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert c.get("/img/nope").status_code == 404


def test_response_validation(env):
    c, rows, _ = env
    seq = c.post("/api/session", json={}).json()["trials"]  # pid 0: practice human, practice AI, ...
    t0, t1 = seq[0], seq[1]
    ok = {"pid": 0, **answer(t0, rows)}
    assert c.post("/api/response", json={**ok, "pid": 5}).status_code == 404
    assert c.post("/api/response", json={**ok, "trial_id": t1["trial_id"]}).status_code == 422
    assert c.post("/api/response", json={**ok, "response": "Atlantis"}).status_code == 422
    assert c.post("/api/response", json={**ok, "chose_rank": 1}).status_code == 422  # human-only
    assert c.post("/api/response", json={**ok, "trial_order": 1}).status_code == 409  # out of order
    assert c.post("/api/response", json={**ok, "rt_ms": -1}).status_code == 422
    assert c.post("/api/response", json=ok).status_code == 200
    assert c.post("/api/response", json=ok).status_code == 409  # double submit
    ai = {"pid": 0, **answer(t1, rows)}
    wrong_card = {**ai, "chose_rank": 2, "response": t1["ai"][0]["country"], "expanded_other": False}
    assert c.post("/api/response", json=wrong_card).status_code == 422
    assert c.post("/api/response", json={**ai, "chose_rank": 0, "expanded_other": False}).status_code == 422
    assert c.post("/api/response", json={**ai, "chose_rank": None}).status_code == 422
    assert c.post("/api/survey", json={"pid": 0, "trust": 4, "usefulness": 4, "geo_knowledge": 4}).status_code == 409


def test_correctness_is_computed_server_side(env):
    c, rows, db = env
    seq = c.post("/api/session", json={}).json()["trials"]
    for t, right in zip(seq[:4], [True, False, False, True]):
        body = answer(t, rows, right) | {"correct": int(not right)}  # client-sent fields are ignored
        assert c.post("/api/response", json={"pid": 0, **body}).status_code == 200
    with sqlite3.connect(db) as conn:
        got = conn.execute("SELECT trial_id, correct, ai_top1_correct, is_practice FROM responses").fetchall()
    assert [g[1] for g in got] == [1, 0, 0, 1]
    assert [g[2] for g in got] == [int(rows[g[0]]["ai_top1"] == rows[g[0]]["country"]) for g in got]
    assert [g[3] for g in got] == [1, 1, 0, 0]


def test_full_flow_and_survey(env):
    c, rows, db = env
    s = c.post("/api/session", json={}).json()
    for i, t in enumerate(s["trials"]):
        assert c.post("/api/response", json={"pid": s["pid"], **answer(t, rows, right=i % 3 == 0)}).status_code == 200
    bad = {"pid": 0, "trust": 8, "usefulness": 5, "geo_knowledge": 2}
    assert c.post("/api/survey", json=bad).status_code == 422
    survey = {"pid": 0, "trust": 5, "usefulness": 6, "geo_knowledge": 2, "free_text": "hi"}
    assert c.post("/api/survey", json=survey).status_code == 200
    assert c.post("/api/survey", json=survey).status_code == 409
    st = c.get("/api/session", params={"pid": 0}).json()
    assert st["done"] == 18 and st["survey_done"]
    with sqlite3.connect(db) as conn:
        resp = pd.read_sql("SELECT * FROM responses", conn)
        surveys = pd.read_sql("SELECT * FROM surveys", conn)
    assert len(resp) == 18 and resp["is_practice"].sum() == 2
    assert list(resp["trial_order"]) == list(range(18))
    assert set(resp.columns) == set(server.RESPONSE_COLS)
    ai = resp[resp.condition == HUMAN_AI]
    assert ai["chose_rank"].notna().all() and resp[resp.condition == HUMAN]["chose_rank"].isna().all()
    assert surveys.to_dict("records")[0] | {"timestamp": None} == {
        "participant_id": 0, "session_id": s["session_id"], "trust": 5, "usefulness": 6, "geo_knowledge": 2,
        "free_text": "hi", "timestamp": None}


def test_static_page_served(env):
    c, _, _ = env
    page = c.get("/")
    assert page.status_code == 200 and "None of these — choose another" in page.text
    assert "The AI can be wrong and sometimes describes things that aren't in the image. Check its evidence." in page.text
