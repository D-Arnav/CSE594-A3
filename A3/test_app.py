"""End-to-end check of both conditions with Gemini stubbed: python test_app.py"""
import base64
import os
import tempfile
from collections import Counter
from itertools import product
from string import ascii_lowercase
from pathlib import Path
from unittest import mock

import db

db.LOCAL_DB = Path(tempfile.mkdtemp()) / "test.db"
os.environ.pop("TURSO_DATABASE_URL", None)

from fastapi.testclient import TestClient

import app as study
import attention
import env
import gemini
import smolvlm

CARDS = [{"country": c, "p": p, "rationale": "A visible clue."}
         for c, p in zip(("Japan", "Germany", "India"), (0.6, 0.3, 0.1))]
OTHER = "United States"  # outside CARDS: choosing it means "none of these"
SURVEY = {"confidence": 4, "mental_demand": 4, "geo_knowledge": 4, "seen_other_condition": False}


def status(response, expected):
    assert response.status_code == expected, (response.status_code, response.text)
    return response.json()


UNIQNAMES = ("".join(letters) for letters in product(ascii_lowercase, repeat=4))


def new_session(client, ai_assist, uniqname=None):
    s = status(client.post("/api/sessions", json={"ai_assist": ai_assist, "uniqname": uniqname or next(UNIQNAMES)}), 200)
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
    images = [Path(url).stem for url in s["images"]]
    live = study.gemini_trials(s["participant_id"])
    assert len(live) == study.N_GEMINI == round(0.2 * study.N_TRIALS) and live <= set(range(study.N_TRIALS))

    # answering before the AI ran is refused
    status(client.post(f"{base}/responses", json={"trial_index": 0, "response": "Japan", "rt_ms": 10}), 409)

    # Gemini trials call Gemini live; the rest show SmolVLM2's A2 output; a refresh shows the logged output again
    with mock.patch.object(gemini, "suggest", return_value=CARDS) as call:
        shown = [status(client.get(f"{base}/ai/{i}"), 200) for i in range(study.N_TRIALS)]
        assert call.call_count == len(live)
        assert [status(client.get(f"{base}/ai/{i}"), 200) for i in range(study.N_TRIALS)] == shown
        assert call.call_count == len(live)
    for i, cards in enumerate(shown):
        assert cards == (CARDS if i in live else smolvlm.suggest(images[i])), i
    status(client.get(f"{base}/ai/999"), 404)

    # every participant's Gemini trials are live calls, never reused from someone else
    other, other_base = new_session(client, True)
    with mock.patch.object(gemini, "suggest", return_value=CARDS) as call:
        for i in range(study.N_TRIALS):
            status(client.get(f"{other_base}/ai/{i}"), 200)
        assert call.call_count == study.N_GEMINI

    # ordering, label space, and retry of a saved answer
    status(client.post(f"{base}/responses", json={"trial_index": 5, "response": "Japan", "rt_ms": 10}), 409)
    status(client.post(f"{base}/responses", json={"trial_index": 0, "response": "Mars", "rt_ms": 10}), 422)
    second = shown[0][1]["country"]
    s = answer_trial(client, base, s, second)
    assert status(client.post(f"{base}/responses", json={"trial_index": 0, "response": shown[0][0]["country"], "rt_ms": 10}), 200)["done"] == 1

    # leaving the AI's suggestions requires expanding; a card chosen after expanding keeps its rank
    outside = next(c for c in gemini.COUNTRIES if c not in {card["country"] for card in shown[1]})
    status(client.post(f"{base}/responses", json={"trial_index": 1, "response": outside, "rt_ms": 10}), 422)
    s = answer_trial(client, base, s, outside, expanded_other=True)
    s = answer_trial(client, base, s, shown[2][0]["country"], expanded_other=True)

    with db.closing(db.connect()) as conn:
        rows = conn.execute("""SELECT response, chose_rank, expanded_other, ai_source, ai_top1 FROM trials
                               WHERE participant_id = ? ORDER BY trial_index""", (s["participant_id"],)).fetchall()
    source = lambda i: "gemini" if i in live else "smolvlm"  # noqa: E731
    assert rows == [(second, 2, 0, source(0), shown[0][0]["country"]),
                    (outside, 0, 1, source(1), shown[1][0]["country"]),
                    (shown[2][0]["country"], 1, 1, source(2), shown[2][0]["country"])], rows

    # a Gemini failure falls back to SmolVLM2 so the participant isn't stuck, and the fallback is logged
    failing, failing_base = new_session(client, True)
    first_live = min(study.gemini_trials(failing["participant_id"]))
    with mock.patch.object(gemini, "suggest", side_effect=gemini.GeminiUnavailable("x")):
        fallback = status(client.get(f"{failing_base}/ai/{first_live}"), 200)
    assert fallback == smolvlm.suggest(Path(failing["images"][first_live]).stem)
    assert db.get_ai_shown(failing["participant_id"], first_live)["source"] == "smolvlm_fallback"
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

    # the check shows a real photo outside this participant's trials, the same one after a refresh
    assert s["attention_check"]["image"] not in s["images"]
    assert status(client.get(base), 200)["attention_check"] == s["attention_check"]
    assert client.get(s["attention_check"]["image"]).status_code == 200
    assert s["attention_check"]["question"] == "Select South Korea."

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
    # starting over with the same uniqname is a new attempt, with photos not seen in the first; entering it again resumes
    uniqname = s["participant_id"].split("-")[0]
    fresh, _ = new_session(client, False, uniqname)
    assert fresh["participant_id"] == f"{uniqname}-noai-2"
    assert not set(fresh["images"]) & set(s["images"])
    assert fresh["attention_check"] is None and not fresh["terminated"] and fresh["done"] == 0
    assert new_session(client, False, uniqname)[0]["participant_id"] == f"{uniqname}-noai-2"


def test_uniqname(client):
    for bad in ("", "ab", "abc1", "abcdefghi", "a b c", "jsmith@gmail.com"):
        status(client.post("/api/sessions", json={"ai_assist": True, "uniqname": bad}), 422)
    status(client.post("/api/sessions", json={"ai_assist": True}), 422)

    # normalized, one session per uniqname and condition, resumed by entering it again
    s, base = new_session(client, True, "  JSmith@UMich.edu ")
    assert s["participant_id"] == "jsmith-ai" and s["ai_assist"]
    with mock.patch.object(gemini, "suggest", return_value=CARDS):
        status(client.get(f"{base}/ai/0"), 200)
    answer_trial(client, base, s)
    again, _ = new_session(client, True, "jsmith")
    assert again["participant_id"] == "jsmith-ai" and again["done"] == 1
    other, _ = new_session(client, False, "jsmith")
    assert other["participant_id"] == "jsmith-noai" and not other["ai_assist"] and other["done"] == 0


def test_photo_assignment(client):
    # a person's two conditions never share a photo, in either order, including attention-check photos
    for first in (True, False):
        uniqname = next(UNIQNAMES)
        a, a_base = new_session(client, first, uniqname)
        b, _ = new_session(client, not first, uniqname)
        photos = lambda s: set(s["images"]) | {study.image_url(i) for i in db.get_assignment(s["participant_id"])["check_images"]}  # noqa: E731
        assert len(photos(a)) == len(photos(b)) == study.N_TRIALS + len(attention.CHECKS)
        assert not photos(a) & photos(b)

        # a partly done session resumes with the same photos and progress
        for _ in range(3):
            response = "Japan"
            if first:
                with mock.patch.object(gemini, "suggest", return_value=CARDS):
                    response = status(client.get(f"{a_base}/ai/{a['done']}"), 200)[0]["country"]
            a = answer_trial(client, a_base, a, response)
        again, _ = new_session(client, first, uniqname)
        assert again["participant_id"] == a["participant_id"] and again["done"] == 3 and again["images"] == a["images"]

    # a finished session can't be redone: entering the uniqname again returns the same finished session
    s, base = new_session(client, False)
    while s["done"] < study.N_TRIALS:
        s = answer_check(client, base, s) if s["attention_check"] else answer_trial(client, base, s)
    status(client.post(f"{base}/survey", json=SURVEY), 200)
    again, _ = new_session(client, False, s["participant_id"].split("-")[0])
    assert again["participant_id"] == s["participant_id"] and again["done"] == study.N_TRIALS and again["survey_done"]
    status(client.post(f"{base}/responses", json={"trial_index": study.N_TRIALS, "response": "Japan", "rt_ms": 10}), 409)


def test_random_trials(client):
    draws = [new_session(client, True)[0] for _ in range(5)]
    for s in draws:
        ids = [Path(image).stem for image in s["images"]]
        assert len(ids) == len(set(ids)) == study.N_TRIALS and set(ids) <= set(study.TRUTH)  # 40 distinct of the 200
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


def test_smolvlm_outputs():
    assert set(smolvlm.OUTPUTS) == set(study.TRUTH)
    for image_id, cards in smolvlm.OUTPUTS.items():
        countries, ps = [c["country"] for c in cards], [c["p"] for c in cards]
        assert len(set(countries)) == 3 and set(countries) <= set(gemini.COUNTRIES), image_id
        assert ps == sorted(ps, reverse=True) and all(0 <= p <= 1 for p in ps) and all(c["rationale"] for c in cards), image_id


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

        # Gemini receives the same photo file participants see
        sent = client.return_value.interactions.create.call_args.kwargs["input"][1]["data"]
        assert base64.b64decode(sent) == path.read_bytes()

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
        test_uniqname(client)
        test_photo_assignment(client)
        test_random_trials(client)
    test_env_file()
    test_smolvlm_outputs()
    test_validation()
    test_retries()
    print("ok")
