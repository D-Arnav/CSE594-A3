import json
import math
import os

import numpy as np
import pytest

import model
from model import (FALLBACK_RATIONALE, apply_temperature, candidate_order, clean_rationale, fit_temperature,
                   load_temperature, pick_device, predict_record, softmax_with_temperature)
from prompts import render_explain, render_json_baseline, render_rank

CANDS = ["United States", "Canada", "Mexico", "Brazil", "Japan", "South Korea"]
CFG = {"seed": 42, "model": {"model_id": "m", "revision": "r", "max_pixels": 1000},
       "inference": {"prompt_versions": {"rank": "v1", "explain": "v1"}, "top_k": 3,
                     "shuffle_candidates": True, "explain_max_new_tokens": 60}}


# ---- pure logic ------------------------------------------------------------------
def test_softmax_with_temperature():
    lp = {"a": -1.0, "b": -2.0, "c": -4.0}
    p = softmax_with_temperature(lp, 1.0)
    assert math.isclose(sum(p.values()), 1.0)
    assert math.isclose(p["a"], math.exp(-1) / (math.exp(-1) + math.exp(-2) + math.exp(-4)))
    hot = softmax_with_temperature(lp, 5.0)
    assert hot["a"] < p["a"] and hot["c"] > p["c"]  # higher T = flatter
    assert softmax_with_temperature({"a": -1000.0, "b": -1001.0}, 0.5)["a"] > 0.8  # no underflow


def test_fit_temperature_recovers_known_T():
    rng = np.random.default_rng(0)
    T_true, n, K = 2.5, 4000, 20
    keys = [f"c{i}" for i in range(K)]
    z = rng.normal(0, 4, size=(n, K))
    p = np.exp(z / T_true)
    p /= p.sum(1, keepdims=True)
    labels = [keys[rng.choice(K, p=row)] for row in p]
    T, before, after = fit_temperature([dict(zip(keys, row)) for row in z], labels)
    assert abs(T - T_true) / T_true < 0.1
    assert after < before


def test_candidate_order_deterministic():
    a = candidate_order(CANDS, 42, "S001")
    assert a == candidate_order(CANDS, 42, "S001")
    assert sorted(a) == sorted(CANDS) and a is not CANDS
    assert any(candidate_order(CANDS, 42, f"S{i:03d}") != a for i in range(2, 10))


def test_clean_rationale():
    assert clean_rationale("  Left-hand traffic.  Kanji signs!\nAnd a third one. Fourth.") == \
        "Left-hand traffic. Kanji signs!"
    assert clean_rationale("No full stop at the end") == "No full stop at the end"
    assert clean_rationale("Yes. Markings are consistent with U.S. road standards. Third.") == \
        "Yes. Markings are consistent with U.S. road standards."
    assert clean_rationale("Kanji signs. Cars drive on the left and the") == "Kanji signs."  # cut at token cap
    assert clean_rationale("   \n") == FALLBACK_RATIONALE
    assert clean_rationale(None) == FALLBACK_RATIONALE


def test_prompts_render():
    r = render_rank(["Chile", "Japan"])
    assert "Chile\nJapan\nWhich country" in r and "{" not in r
    assert render_explain("Chile").count("Chile") == 2
    j = render_json_baseline(["Chile", "Japan"])
    assert "Chile\nJapan" in j and '{"guesses": [{"country"' in j


def test_apply_temperature_keeps_order():
    rec = {"logprobs": {"a": -0.1, "b": -2.0, "c": -5.0},
           "top3": [{"country": "a", "p": 0.0}, {"country": "b", "p": 0.0}, {"country": "c", "p": 0.0}]}
    apply_temperature(rec, 3.0)
    assert rec["temperature"] == 3.0
    assert [t["country"] for t in rec["top3"]] == ["a", "b", "c"]
    assert rec["top3"][0]["p"] == round(softmax_with_temperature(rec["logprobs"], 3.0)["a"], 2)


def test_load_temperature(tmp_path, monkeypatch):
    f = tmp_path / "temperature.json"
    monkeypatch.setattr(model, "TEMPERATURE_FILE", f)
    assert load_temperature({"inference": {"temperature": 1.3}}) == (1.3, "config")
    f.write_text(json.dumps({"T": 2.0}))
    assert load_temperature({"inference": {"temperature": 1.3}}) == (2.0, "temperature.json")


def test_pick_device_explicit():
    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in {"cuda", "mps", "cpu"}


class FakeGM:
    """Scores depend on the prompt's candidate order, so n_orderings averaging is observable."""
    model_id, revision, max_pixels = "m", "r", 1000

    def __init__(self):
        self.prompts, self.explained = [], []

    def score_candidates(self, image, prompt, candidates, naive=False, answer_format="{c}"):
        self.prompts.append(prompt)
        listed = prompt.split("\n")[1:1 + len(candidates)]
        lp = {c: -1.0 * CANDS.index(c) - 0.01 * listed.index(c) for c in candidates}
        return lp, {c: len(c.split()) + 1 for c in candidates}

    def explain(self, image, country, prompt_version="v1", max_new_tokens=60):
        self.explained.append(country)
        return f"Because {country}."


def test_predict_record_shape():
    gm = FakeGM()
    rec = predict_record(gm, None, "S001", CFG, CANDS, 2.0)
    for k in ["trial_id", "model_id", "revision", "prompt_versions", "temperature", "candidate_order",
              "logprobs", "probs", "n_answer_tokens", "top3", "max_pixels", "seed", "runtime_s"]:
        assert k in rec
    assert rec["candidate_order"] == candidate_order(CANDS, 42, "S001")
    assert [t["country"] for t in rec["top3"]] == ["United States", "Canada", "Mexico"] == gm.explained
    assert rec["top3"][0]["rationale"] == "Because United States."
    assert math.isclose(sum(rec["probs"].values()), 1.0, abs_tol=1e-3)
    assert rec["top3"][0]["p"] == round(softmax_with_temperature(rec["logprobs"], 2.0)["United States"], 2)
    assert rec["n_answer_tokens"]["South Korea"] == 3
    json.dumps(rec)  # serializable
    # same key -> identical scores; key overrides trial_id for ordering
    assert predict_record(FakeGM(), None, "X", CFG, CANDS, 2.0, key="S001")["candidate_order"] == rec["candidate_order"]


def test_predict_record_averages_orderings():
    cfg = {**CFG, "inference": {**CFG["inference"], "n_orderings": 3}}
    gm = FakeGM()
    rec = predict_record(gm, None, "S001", cfg, CANDS, 1.0)
    assert len(gm.prompts) == 3 and len(rec["candidate_orders"]) == 3
    orders = rec["candidate_orders"]
    want = -1.0 - 0.01 * sum(o.index("Canada") for o in orders) / 3
    assert math.isclose(rec["logprobs"]["Canada"], round(want, 4))


# ---- real model (RUN_MODEL_TESTS=1); GEO_TEST_DEVICE=cpu runs it in float32 -------------------
@pytest.fixture(scope="module")
def gm():
    from common import load_config
    dev = os.environ.get("GEO_TEST_DEVICE")
    return model.GeoModel(load_config(), device=dev, dtype="float32" if dev == "cpu" else None)


@pytest.fixture(scope="module")
def image():
    from PIL import Image
    if os.environ.get("GEO_TEST_IMAGE"):
        return Image.open(os.environ["GEO_TEST_IMAGE"]).convert("RGB")
    x = np.linspace(0, 255, 256, dtype=np.uint8)
    return Image.fromarray(np.stack([*np.meshgrid(x, x[::-1]), np.full((256, 256), 90, np.uint8)], -1))


@pytest.mark.model
def test_answer_tokens_end_of_turn_and_prefix(gm, image):
    prompt = render_rank(CANDS)
    tail = gm.answer_tokens(prompt, "South Korea")
    tok = gm.tokenizer
    assert tok.decode(tail[:-1]) == "South Korea"
    assert tail[-1] in tok.all_special_ids  # end-of-turn (<|im_end|> for Qwen) is scored
    prefix = gm._encode(gm.build_messages(image, prompt), add_generation_prompt=True)["input_ids"][0].tolist()
    full = gm._encode(gm.build_messages(image, prompt, "South Korea"))["input_ids"][0].tolist()
    assert full[:len(prefix)] == prefix
    assert full[len(prefix):len(prefix) + len(tail)] == tail
    assert not tok.decode(full[len(prefix) + len(tail):]).strip()  # only whitespace after end-of-turn


@pytest.mark.model
def test_cached_matches_naive(gm, image):
    prompt = render_rank(candidate_order(CANDS, 42, "t"))
    cached, n1 = gm.score_candidates(image, prompt, CANDS)
    naive, n2 = gm.score_candidates(image, prompt, CANDS, naive=True)
    assert n1 == n2
    tol = 1e-3 if gm.dtype == __import__("torch").float32 else 5e-2
    diffs = {c: abs(cached[c] - naive[c]) for c in CANDS}
    assert max(diffs.values()) < tol, diffs


@pytest.mark.model
def test_predict_record_real(gm, image):
    rec = predict_record(gm, image, "S001", CFG, CANDS, 1.0)
    assert len(rec["top3"]) == 3 and all(t["rationale"] for t in rec["top3"])
    assert set(rec["logprobs"]) == set(CANDS) and all(v <= 0 for v in rec["logprobs"].values())
    assert rec["model_id"] == gm.model_id and rec["max_pixels"] == gm.max_pixels
