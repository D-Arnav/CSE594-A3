"""End-to-end check of both conditions with Gemini stubbed: python test_app.py"""
import base64
import io
import os
import tempfile
from collections import Counter
from pathlib import Path
from unittest import mock

from PIL import Image

import db

db.LOCAL_DB = Path(tempfile.mkdtemp()) / "test.db"
os.environ.pop("TURSO_DATABASE_URL", None)

from fastapi.testclient import TestClient

import app as study
import attention
import env
import gemini

CARDS = [{"country": c, "p": p, "rationale": "A visible clue."}
         for c, p in zip(("Japan", "Germany", "India"), (0.6, 0.3, 0.1))]
OTHER = "United States"  # outside CARDS: choosing it means "none of these"
SURVEY = {"confidence": 4, "mental_demand": 4, "geo_knowledge": 4, "seen_other_condition": False}


def status(response, expected):
    assert response.status_code == expected, (response.status_code, response.text)
    return response.json()


def new_session(client, ai_assist):
    s = status(client.post("/api/sessions", json={"ai_assist": ai_assist}), 200)
    return s, f"/api/sessions/{s['participant_id']}"


def answer_check(client, base, s, correct=True):
    check = attention.CHECKS[s["attention_check"]["index"]]
    answer = check["answer"] if correct else next(o for o in check["options"] if o != check["answer"])
    return status(client.post(f"{base}/attention", json={"check_index": s["attention_check"]["index"], "answer": answer}), 200)


def answer_trial(client, base, s, response="Japan", **extra):
    return status(client.post(f"{base}/responses", json={"trial_index": s["done"], "response": response, "rt_ms": 10, **extra}), 200)


def test_no_ai_flow(client):
    s, base = new_session(client, False)
    status(client.get(f"{base}/ai/0"), 403)
    status(client.post(f"{base}/survey", json=SURVEY), 409)
    while s["done"] < study.N_TRIALS:
        s = answer_check(client, base, s) if s["attention_check"] else answer_trial(client, base, s)
    assert s["checks_done"] == len(attention.CHECKS) and not s["check_failures"]
    status(client.post(f"{base}/survey", json={**SURVEY, "trust": 4, "usefulness": 4}), 422)
    assert status(client.post(f"{base}/survey", json=SURVEY), 200)["survey_done"]


def test_ai_flow(client):
    s, base = new_session(client, True)

    # answering before the AI ran is refused; one Gemini call per photo across participants
    status(client.post(f"{base}/responses", json={"trial_index": 0, "response": "Japan", "rt_ms": 10}), 409)
    with mock.patch.object(gemini, "suggest", return_value=CARDS) as call:
        assert status(client.get(f"{base}/ai/0"), 200) == CARDS
        status(client.get(f"{base}/ai/0"), 200)
        assert call.call_count == 1
    status(client.get(f"{base}/ai/999"), 404)

    # ordering, label space, and retry of a saved answer
    status(client.post(f"{base}/responses", json={"trial_index": 5, "response": "Japan", "rt_ms": 10}), 409)
    status(client.post(f"{base}/responses", json={"trial_index": 0, "response": "Mars", "rt_ms": 10}), 422)
    s = answer_trial(client, base, s, "Germany")
    assert status(client.post(f"{base}/responses", json={"trial_index": 0, "response": "India", "rt_ms": 10}), 200)["done"] == 1

    # leaving the AI's suggestions requires expanding; a card chosen after expanding keeps its rank
    with mock.patch.object(gemini, "suggest", return_value=CARDS):
        status(client.get(f"{base}/ai/1"), 200)
        status(client.get(f"{base}/ai/2"), 200)
    status(client.post(f"{base}/responses", json={"trial_index": 1, "response": OTHER, "rt_ms": 10}), 422)
    s = answer_trial(client, base, s, OTHER, expanded_other=True)
    s = answer_trial(client, base, s, "Japan", expanded_other=True)

    with db.closing(db.connect()) as conn:
        rows = conn.execute("""SELECT response, chose_rank, expanded_other, ai_top1 FROM responses
                               WHERE participant_id = ? ORDER BY trial_index""", (s["participant_id"],)).fetchall()
    assert rows == [("Germany", 2, 0, "Japan"), (OTHER, 0, 1, "Japan"), ("Japan", 1, 1, "Japan")], rows

    # Gemini failure surfaces as a retryable 503
    with mock.patch.object(gemini, "suggest", side_effect=gemini.GeminiUnavailable("x")):
        status(client.get(f"{base}/ai/3"), 503)
    status(client.post(f"{base}/survey", json={**SURVEY, "trust": 4, "usefulness": 4}), 409)


def test_attention(client):
    s, base = new_session(client, False)

    # no check before the first photo, and none can be answered before it's due
    assert s["attention_check"] is None
    status(client.post(f"{base}/attention", json={"check_index": 0, "answer": "South Korea"}), 409)

    # a due check blocks its trial, and its answer never reaches the browser
    while not s["attention_check"]:
        s = answer_trial(client, base, s)
    assert s["done"] == attention.CHECKS[0]["after_trial"] and "answer" not in s["attention_check"]
    status(client.post(f"{base}/responses", json={"trial_index": s["done"], "response": "Japan", "rt_ms": 10}), 409)
    status(client.post(f"{base}/attention", json={"check_index": 1, "answer": "Germany"}), 409)
    status(client.post(f"{base}/attention", json={"check_index": 0, "answer": "Mars"}), 422)

    # one failure warns and continues; the retry of a saved answer changes nothing
    first = answer_check(client, base, s, correct=False)
    assert first["passed"] is False and not first["terminated"] and first["attention_check"] is None
    right = attention.CHECKS[0]["answer"]
    retry = status(client.post(f"{base}/attention", json={"check_index": 0, "answer": right}), 200)
    assert retry["passed"] is None and retry["check_failures"] == 1

    # the next check blocks its trial too
    s = first
    while not s["attention_check"]:
        s = answer_trial(client, base, s)
    assert s["done"] == attention.CHECKS[1]["after_trial"]
    status(client.post(f"{base}/responses", json={"trial_index": s["done"], "response": "Japan", "rt_ms": 10}), 409)

    # a second failure ends the session everywhere, and a new session starts clean
    s = answer_check(client, base, s, correct=False)
    assert s["passed"] is False and s["terminated"]
    assert status(client.get(base), 200)["terminated"]
    status(client.post(f"{base}/responses", json={"trial_index": s["done"], "response": "Japan", "rt_ms": 10}), 409)
    status(client.post(f"{base}/attention", json={"check_index": 2, "answer": "Germany"}), 409)
    status(client.post(f"{base}/survey", json=SURVEY), 409)
    fresh, _ = new_session(client, False)
    assert fresh["attention_check"] is None and not fresh["terminated"] and fresh["done"] == 0


def test_random_trials(client):
    draws = [new_session(client, True)[0] for _ in range(5)]
    for s in draws:
        ids = [Path(image).stem for image in s["images"]]
        assert len(ids) == len(set(ids)) == study.N_TRIALS
        assert Counter(study.TRUTH[i] for i in ids) == Counter({c: study.TRIALS_PER_COUNTRY for c in gemini.COUNTRIES})
        assert status(client.get(f"/api/sessions/{s['participant_id']}"), 200)["images"] == s["images"]  # stable on refresh
    assert len({tuple(s["images"]) for s in draws}) == len(draws)  # differs across participants
    assert len({frozenset(s["images"]) for s in draws}) == len(draws)

    # answers never leak
    assert "country" not in draws[0]
    assert client.get("/images.csv").status_code == 404
    assert client.get("/images/../images.csv").status_code == 404


def test_env_file():
    path = Path(tempfile.mkdtemp()) / ".env"
    path.write_text('# COMMENTED=no\nTEST_PLAIN=abc=123\nTEST_QUOTED = "xyz"\nTEST_SET=from-file\nnot a setting\n')
    with mock.patch.dict(os.environ, {"TEST_SET": "from-environment"}):
        env.load(path)
        assert (os.environ["TEST_PLAIN"], os.environ["TEST_QUOTED"]) == ("abc=123", "xyz")
        assert os.environ["TEST_SET"] == "from-environment" and "COMMENTED" not in os.environ
    env.load(path.with_name("missing.env"))


def test_validation():
    good = {"suggestions": CARDS}
    assert [s.country for s in gemini.Suggestions.model_validate(good).suggestions] == ["Japan", "Germany", "India"]
    for bad in (CARDS[:2], [CARDS[0]] * 3, [{**CARDS[0], "country": "France"}, *CARDS[1:]], [{**CARDS[0], "p": 2}, *CARDS[1:]]):
        try:
            gemini.Suggestions.model_validate({"suggestions": bad})
        except ValueError:  # pydantic's ValidationError
            continue
        raise AssertionError(bad)


def test_retries():
    bad, ok = mock.Mock(output_text="not json"), mock.Mock(output_text=gemini.Suggestions(suggestions=CARDS).model_dump_json())
    path = study.IMAGE_DIR / f"{next(iter(study.TRUTH))}.jpg"
    with mock.patch.object(gemini.time, "sleep"), mock.patch.object(gemini, "client") as client:
        client.return_value.interactions.create.side_effect = [bad, ok]
        assert gemini.suggest(path)[0]["country"] == "Japan"

        # the AI receives the photo at 1/8 scale; the file participants see is untouched
        sent = client.return_value.interactions.create.call_args.kwargs["input"][1]["data"]
        with Image.open(io.BytesIO(base64.b64decode(sent))) as small, Image.open(path) as original:
            assert small.size == (original.width // gemini.DOWNSCALE, original.height // gemini.DOWNSCALE)

        client.return_value.interactions.create.side_effect = [bad] * gemini.ATTEMPTS
        try:
            gemini.suggest(path)
        except gemini.GeminiUnavailable:
            return
    raise AssertionError("expected GeminiUnavailable")


if __name__ == "__main__":
    with TestClient(study.app) as client:
        test_no_ai_flow(client)
        test_ai_flow(client)
        test_attention(client)
        test_random_trials(client)
    test_env_file()
    test_validation()
    test_retries()
    print("ok")
