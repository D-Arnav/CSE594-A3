import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import model
import server
from common import OUTPUTS, ROOT, countries, load_config
from model import candidate_order, image_key


class FakeGM:
    model_id, revision, max_pixels = "fake-m", "fake-r", 1000

    def __init__(self, cfg):
        pass

    def score_candidates(self, image, prompt, candidates, naive=False, answer_format="{c}"):
        favored = "Canada" if image.getpixel((0, 0))[0] > 100 else "Japan"
        return {c: (0.0 if c == favored else -2.0 - 0.1 * i) for i, c in enumerate(candidates)}, \
            {c: 2 for c in candidates}

    def explain(self, image, country, prompt_version="v1", max_new_tokens=60):
        return f"Clue for {country}."


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (color, 0, 0)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "TEMPERATURE_FILE", tmp_path / "none.json")  # -> config temperature
    for split, tid, color in [("study", "S001", 200), ("dev", "D001", 10)]:
        d = tmp_path / split
        d.mkdir()
        (d / f"{tid}.jpg").write_bytes(jpeg(color))
        (d / f"{split}_meta.csv").write_text(f"trial_id,image_path,country\n{tid},{d / f'{tid}.jpg'},Canada\n"
                                             f"{tid[0]}999,{d / 'missing.jpg'},Canada\n")
    cfg = load_config()
    cfg["server"] = {"max_upload_mb": 0.01}
    app = server.create_app(cfg, model_factory=FakeGM, data_dir=tmp_path)
    with TestClient(app) as client:
        yield client, cfg


def test_health(setup):
    client, _ = setup
    assert client.get("/health").json() == {"status": "ok", "model_id": "fake-m", "revision": "fake-r"}


def test_predict_trial_id_record_shape(setup):
    client, cfg = setup
    r = client.post("/predict", json={"trial_id": "S001"})
    assert r.status_code == 200
    rec = r.json()
    labels = countries(cfg)
    assert rec["trial_id"] == "S001" and rec["model_id"] == "fake-m"
    assert rec["candidate_order"] == candidate_order(labels, cfg["seed"], "S001")
    assert set(rec["logprobs"]) == set(labels) and set(rec["n_answer_tokens"]) == set(labels)
    assert [t["country"] for t in rec["top3"]][0] == "Canada"
    assert rec["top3"][0]["rationale"] == "Clue for Canada."
    assert rec["temperature"] == cfg["inference"]["temperature"] and rec["temperature_source"] == "config"
    assert abs(sum(rec["probs"].values()) - 1) < 1e-3
    for k in ["prompt_versions", "max_pixels", "seed", "runtime_s"]:
        assert k in rec
    assert client.post("/predict", json={"trial_id": "D001"}).json()["top3"][0]["country"] == "Japan"


def test_predict_upload_matches_hash_key(setup):
    client, cfg = setup
    data = jpeg(200)
    rec = client.post("/predict", files={"image": ("x.jpg", data, "image/jpeg")}).json()
    assert rec["trial_id"] == "upload"
    assert rec["candidate_order"] == candidate_order(countries(cfg), cfg["seed"], image_key(data))
    assert rec["top3"][0]["country"] == "Canada"


@pytest.mark.parametrize("kwargs, code", [
    ({"json": {"trial_id": "S404"}}, 404),
    ({"json": {"trial_id": "S999"}}, 404),  # in meta, image file missing
    ({"json": {"foo": 1}}, 400),
    ({"json": ["S001"]}, 400),
    ({"content": b"{not json", "headers": {"content-type": "application/json"}}, 400),
    ({"files": {"image": ("x.jpg", b"not an image", "image/jpeg")}}, 400),
    ({"files": {"other": ("x.jpg", jpeg(10), "image/jpeg")}}, 400),
    ({"files": {"image": ("x.bmp", b"\0" * 20000, "image/bmp")}}, 413),
    ({"content": b"hello", "headers": {"content-type": "text/plain"}}, 415),
])
def test_predict_validation(setup, kwargs, code):
    client, _ = setup
    assert client.post("/predict", **kwargs).status_code == code


def test_cors_header(setup):
    client, _ = setup
    r = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_resolve_trial(tmp_path):
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "dev_meta.csv").write_text("trial_id,image_path\nD007,data/dev/D007.jpg\n")
    assert server.resolve_trial("D007", tmp_path) == "data/dev/D007.jpg"
    assert server.resolve_trial("S007", tmp_path) is None


# ---- SPEC 10: live server must reproduce the cached outputs -----------------------
def check_parity(client, cached, lp_tol=0.05, p_tol=0.02):
    """POST each cached record's trial_id and compare with the cached record. -> seconds per image."""
    import time
    secs = []
    for want in cached:
        t0 = time.time()
        r = client.post("/predict", json={"trial_id": want["trial_id"]})
        secs.append(time.time() - t0)
        assert r.status_code == 200, r.text
        got = r.json()
        assert [t["country"] for t in got["top3"]] == [t["country"] for t in want["top3"]], want["trial_id"]
        assert got["candidate_order"] == want["candidate_order"]
        for c, lp in want["logprobs"].items():
            assert abs(got["logprobs"][c] - lp) <= lp_tol, (want["trial_id"], c, got["logprobs"][c], lp)
        if got["temperature"] == want["temperature"]:
            for g, w in zip(got["top3"], want["top3"]):
                assert abs(g["p"] - w["p"]) <= p_tol, (want["trial_id"], g, w)
    return secs


@pytest.mark.model
def test_parity_with_cached_outputs():
    path = OUTPUTS / "raw_outputs.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not found; run predict.py --split study first")
    cached = []
    for line in filter(str.strip, path.read_text().splitlines()):
        rec = json.loads(line)
        rel = server.resolve_trial(rec["trial_id"])
        if rel and (ROOT / rel).exists():
            cached.append(rec)
        if len(cached) == 5:
            break
    if not cached:
        pytest.skip("no cached trial has a resolvable image in data/study or data/dev")
    with TestClient(server.create_app()) as client:
        secs = check_parity(client, cached)
    print("seconds per image:", [round(s, 2) for s in secs])
