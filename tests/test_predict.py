import json
import sys

import pandas as pd
import pytest

import model
import predict
from predict import (COLUMNS, baseline_record, baseline_summary, dataset_row, gt_rank, load_existing,
                     normalize_country, parse_guesses, prompt_version_str)

LABELS = ["United States", "Canada", "Mexico", "Japan", "South Korea"]
CFG = {"seed": 42, "model": {"model_id": "m", "revision": "r", "max_pixels": 1000},
       "inference": {"temperature": 1.0, "prompt_versions": {"rank": "v1", "explain": "v1"}, "top_k": 3,
                     "shuffle_candidates": True, "explain_max_new_tokens": 60}}


def rec(tid, lp, model_id="m", revision="r", pv=None):
    ranked = sorted(lp, key=lambda c: -lp[c])
    return {"trial_id": tid, "model_id": model_id, "revision": revision,
            "prompt_versions": pv or {"rank": "v1", "explain": "v1"}, "temperature": 1.0, "logprobs": lp,
            "top3": [{"country": c, "p": 0.1, "rationale": f"r {c}"} for c in ranked[:3]], "max_pixels": 1000}


META = {"trial_id": "S001", "image_path": "data/study/S001.jpg", "source_id": "123", "country": "Canada",
        "cluster": "North America", "has_text": True}


def test_gt_rank():
    lp = {"United States": -0.5, "Canada": -1.0, "Mexico": -3.0}
    assert gt_rank(lp, "United States") == 1 and gt_rank(lp, "Canada") == 2 and gt_rank(lp, "Mexico") == 3
    assert gt_rank(lp, "Japan") == ""
    assert gt_rank({"a": -1.0, "b": -1.0}, "a") == 1  # ties keep candidate order, like top-3


def test_dataset_row():
    row = dataset_row(rec("S001", {"United States": -0.5, "Canada": -1.0, "Mexico": -3.0, "Japan": -4.0}), META)
    assert list(row) == COLUMNS
    assert row["ai_top1"] == "United States" and row["ai_top2"] == "Canada" and row["ai_rationale3"] == "r Mexico"
    assert row["ai_correct_top1"] is False and row["ai_correct_top3"] is True and row["gt_rank"] == 2
    assert row["error_category"] == row["human_catchable"] == row["rationale_faithful_top1"] == ""
    assert row["prompt_version"] == "rank=v1;explain=v1" == prompt_version_str({"rank": "v1", "explain": "v1"})


def test_load_existing_resume(tmp_path):
    p = tmp_path / "raw.jsonl"
    assert load_existing(p, CFG) == []
    p.write_text(json.dumps(rec("S001", {"Canada": -1.0})) + "\n")
    assert [r["trial_id"] for r in load_existing(p, CFG)] == ["S001"]
    with p.open("a") as f:
        f.write(json.dumps(rec("S002", {"Canada": -1.0}, pv={"rank": "v2", "explain": "v1"})) + "\n")
    with pytest.raises(SystemExit, match="--overwrite"):
        load_existing(p, CFG)
    assert load_existing(p, CFG, overwrite=True) == [] and not p.exists()


@pytest.mark.parametrize("text,countries", [
    ('{"guesses": [{"country": "Japan", "confidence": 0.7, "reasoning": "kanji"}]}', ["Japan"]),
    ('```json\n{"guesses": [{"country": "USA", "confidence": "0.5"}, {"country": "Canada"}]}\n```',
     ["USA", "Canada"]),
    ('Sure! Here is my answer: [{"country": "Mexico", "confidence": 0.4}] Hope that helps.', ["Mexico"]),
    ('{"country": "Japan", "confidence": 0.9}', ["Japan"]),
    ('{"guesses": [{"country": "Japan", "confidence": 0.7},]}', None),  # trailing comma = invalid JSON
    ("I think it is Japan.", None),
    ("", None),
    ('{"guesses": "Japan"}', None),
])
def test_parse_guesses(text, countries):
    g = parse_guesses(text)
    assert (None if g is None else [x["country"] for x in g]) == countries


def test_normalize_country():
    assert normalize_country("japan", LABELS) == "Japan"
    assert normalize_country(" USA ", LABELS) == "United States"
    assert normalize_country("Korea", LABELS) == "South Korea"
    assert normalize_country("United Kingdom", LABELS) is None  # not in this label set
    assert normalize_country("Atlantis", LABELS) is None and normalize_country(None, LABELS) is None


def test_baseline_summary():
    recs = [
        baseline_record("D1", "Japan", '{"guesses": [{"country": "Japan", "confidence": 0.6}, '
                        '{"country": "Korea", "confidence": 0.3}, {"country": "China", "confidence": 0.1}]}',
                        LABELS, 1.0),
        baseline_record("D2", "Canada", '{"guesses": [{"country": "USA", "confidence": 0.9}, '
                        '{"country": "Canada", "confidence": 0.9}]}', LABELS, 1.0),
        baseline_record("D3", "Mexico", '{"guesses": [{"country": "Mex', LABELS, 1.0, truncated=True),
    ]
    s = baseline_summary(recs)
    assert s["n"] == 3 and s["n_parsed"] == 2 and s["parse_failure_rate"] == round(1 / 3, 4)
    assert s["truncated_rate"] == round(1 / 3, 4)
    assert s["out_of_label_set_rate"] == 0.2  # China
    assert s["not_three_guesses_rate"] == 0.5
    assert s["confidence_sum_mean"] == 1.4 and s["confidence_sum_above_1.05"] == 0.5
    assert s["top1_acc_parsed"] == 0.5 and s["top3_acc_parsed"] == 1.0
    assert s["top1_acc_overall"] == round(1 / 3, 4) and s["top3_acc_overall"] == round(2 / 3, 4)


class FakeGM:
    model_id, revision, max_pixels = "m", "r", 1000
    device, dtype = "cpu", "float32"
    calls = 0

    def __init__(self, cfg):
        pass

    def score_candidates(self, image, prompt, candidates, naive=False, answer_format="{c}"):
        FakeGM.calls += 1
        favored = "Canada" if image.getpixel((0, 0))[0] > 100 else "Japan"
        return {c: (0.0 if c == favored else -2.0 - 0.1 * i) for i, c in enumerate(candidates)}, \
            {c: 2 for c in candidates}

    def explain(self, image, country, prompt_version="v1", max_new_tokens=60):
        return f"Clue for {country}."


def test_main_end_to_end(tmp_path, monkeypatch):
    from PIL import Image
    (tmp_path / "data" / "dev").mkdir(parents=True)
    rows = []
    for i, (country, color) in enumerate([("Canada", 200), ("Japan", 10), ("Canada", 10)], 1):
        tid = f"D{i:03d}"
        Image.new("RGB", (8, 8), (color, 0, 0)).save(tmp_path / "data" / "dev" / f"{tid}.jpg")
        rows.append({"trial_id": tid, "image_path": f"data/dev/{tid}.jpg", "source_id": f"{i}", "country": country,
                     "cluster": "x", "lat": 0, "lon": 0, "has_text": False, "ocr_text": "", "split": "dev"})
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "dev" / "dev_meta.csv", index=False)
    (tmp_path / "config.yaml").write_text(json.dumps({**CFG, "data": {"countries": LABELS}}))
    tfile = tmp_path / "outputs" / "temperature.json"
    for mod, name, val in [(predict, "DATA", tmp_path / "data"), (predict, "OUTPUTS", tmp_path / "outputs"),
                           (predict, "ROOT", tmp_path), (predict, "TEMPERATURE_FILE", tfile),
                           (model, "TEMPERATURE_FILE", tfile), (model, "GeoModel", FakeGM),
                           (predict, "countries", lambda cfg: LABELS)]:
        monkeypatch.setattr(mod, name, val)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["predict.py", "--split", "dev", "--config", str(tmp_path / "config.yaml"),
                                          *args])
        predict.main()

    run("--limit", "2")
    assert FakeGM.calls == 2
    run("--fit-temperature")  # resumes: only D003 is new
    assert FakeGM.calls == 3
    T = json.loads(tfile.read_text())
    assert T["n"] == 3 and T["split"] == "dev" and T["nll_after"] <= T["nll_before"]
    raw = predict.read_jsonl(tmp_path / "outputs" / "dev" / "raw_outputs.jsonl")
    assert [r["trial_id"] for r in raw] == ["D001", "D002", "D003"]
    assert all(r["temperature"] == T["T"] and r["temperature_source"] == "temperature.json" for r in raw)
    df = pd.read_csv(tmp_path / "outputs" / "dev" / "dev_dataset.csv")
    assert list(df.columns) == COLUMNS and len(df) == 3
    assert df["ai_correct_top1"].tolist() == [True, True, False] and df["gt_rank"].tolist()[:2] == [1, 1]
    assert df["ai_rationale1"][0] == "Clue for Canada."
