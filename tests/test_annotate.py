from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import annotate as A

SRC = Path(A.__file__)


def make_pred(n_err=6, n_ok=10, img_dir=None):
    rows = []
    for i in range(n_err + n_ok):
        tid = f"S{i + 1:03d}"
        wrong = i < n_err
        img = ""
        if img_dir is not None:
            img = str(Path(img_dir) / f"{tid}.jpg")
            Image.new("RGB", (32, 32), (i * 10, 100, 50)).save(img)
        rows.append({
            "trial_id": tid, "image_path": img, "source_id": str(i), "country": "Chile",
            "cluster": "South America", "has_text": False,
            "ai_top1": "Argentina" if wrong else "Chile", "ai_top2": "Chile" if wrong else "Argentina",
            "ai_top3": "Spain", "ai_p1": 0.7 if i == 0 else 0.4, "ai_p2": 0.3, "ai_p3": 0.1,
            "ai_rationale1": "r1", "ai_rationale2": "r2", "ai_rationale3": "r3",
        })
    return pd.DataFrame(rows)


def test_queue_all_errors_plus_n_correct_deterministic():
    pred = make_pred(6, 10)
    q = A.build_queue(pred, n_correct=4, seed=1)
    assert q[:6] == [f"S{i:03d}" for i in range(1, 7)]
    assert len(q) == 10 and len(set(q)) == 10
    assert A.build_queue(pred, n_correct=4, seed=1) == q
    assert any(A.build_queue(pred, n_correct=4, seed=s) != q for s in range(2, 10))
    # fewer correct trials than requested -> all of them
    assert len(A.build_queue(pred, n_correct=50, seed=1)) == 16


def test_queue_subset_is_seeded_subset_of_full_queue():
    pred = make_pred(20, 30)
    full = A.build_queue(pred, n_correct=20, seed=3)
    sub = A.build_queue(pred, n_correct=20, seed=3, subset_frac=0.25)
    assert len(sub) == 10 and set(sub) <= set(full)
    assert [t for t in full if t in sub] == sub  # same order
    assert A.build_queue(pred, n_correct=20, seed=3, subset_frac=0.25) == sub


def test_validate():
    ok_err = {"error_category": "HAL", "rationale_faithful_top1": "hallucinated", "human_catchable": "yes"}
    assert A.validate(ok_err, error=True) == []
    assert A.validate({**ok_err, "error_category": None}, error=True)
    assert A.validate({**ok_err, "human_catchable": ""}, error=True)
    assert A.validate({**ok_err, "error_category": "OVR"}, error=True)
    assert A.validate({**ok_err, "rationale_faithful_top1": None}, error=True)
    assert A.validate({"rationale_faithful_top1": "faithful"}, error=False) == []
    assert A.validate({"rationale_faithful_top1": "faithful", "error_category": "HAL"}, error=False)
    assert A.validate({"rationale_faithful_top1": "faithful", "human_catchable": "no"}, error=False)


def test_upsert_idempotent_and_keyed_by_rater(tmp_path):
    out = tmp_path / "ann.csv"
    row = {"trial_id": "S001", "rater": "a", "error_category": "TXT",
           "rationale_faithful_top1": "partially", "human_catchable": "maybe", "note": "x"}
    A.upsert(out, row)
    A.upsert(out, row)
    A.upsert(out, {**row, "error_category": "NBR"})
    A.upsert(out, {**row, "rater": "b"})
    A.upsert(out, {"trial_id": "S002", "rater": "a", "rationale_faithful_top1": "faithful"})
    ann = A.load_annotations(out)
    assert list(ann.columns) == A.ANN_COLUMNS
    assert len(ann) == 3
    a1 = ann[(ann.trial_id == "S001") & (ann.rater == "a")].iloc[0]
    assert a1.error_category == "NBR"
    assert pd.Timestamp(a1.timestamp).tzinfo is not None
    s2 = ann[ann.trial_id == "S002"].iloc[0]
    assert s2.error_category == "" and s2.human_catchable == ""


def test_resume_progress_and_first_unannotated(tmp_path):
    out = tmp_path / "ann.csv"
    queue = ["S001", "S002", "S003"]
    ann = A.load_annotations(out)
    assert A.progress(queue, ann, "a") == (0, 3) and A.first_unannotated(queue, ann, "a") == 0
    A.upsert(out, {"trial_id": "S001", "rater": "a", "rationale_faithful_top1": "none"})
    A.upsert(out, {"trial_id": "S002", "rater": "b", "rationale_faithful_top1": "none"})
    ann = A.load_annotations(out)  # reload from disk = resume
    assert A.progress(queue, ann, "a") == (1, 3)
    assert A.first_unannotated(queue, ann, "a") == 1
    assert A.first_unannotated(queue, ann, "b") == 0
    for t in queue:
        A.upsert(out, {"trial_id": t, "rater": "a", "rationale_faithful_top1": "none"})
    assert A.first_unannotated(queue, A.load_annotations(out), "a") == 3


def test_parse_args_tolerates_streamlit_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["streamlit", "run", "src/annotate.py", "--",
                                     "--rater", "r2", "--subset-frac", "0.25", "--bogus", "1"])
    args = A.parse_args()
    assert args.rater == "r2" and args.subset_frac == 0.25 and args.n_correct == 50
    assert isinstance(args.seed, int)


def test_app_renders_and_saves(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    pred_path = tmp_path / "pred.csv"
    out = tmp_path / "ann.csv"
    make_pred(2, 2, img_dir=tmp_path).to_csv(pred_path, index=False)
    for k, v in {"PRED": pred_path, "OUT": out, "RATER": "t", "N_CORRECT": 1, "SEED": 0}.items():
        monkeypatch.setenv(f"ANNOTATE_{k}", str(v))

    at = AppTest.from_file(str(SRC), default_timeout=30).run()
    assert not at.exception
    assert any("S001" in s.value for s in at.subheader)
    assert any("OVR" in e.value for e in at.error)  # S001 has p1 = 0.7 and is wrong

    # incomplete save -> validation errors, nothing written
    at.button(key="save").click().run()
    assert not out.exists()

    at.radio(key="cat_S001").set_value("HAL")
    at.radio(key="faith_S001").set_value("hallucinated")
    at.radio(key="catch_S001").set_value("yes")
    at.text_input(key="note_S001").input("sign is fake")
    at.button(key="save").click().run()
    assert not at.exception
    ann = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert ann.iloc[0][["trial_id", "rater", "error_category", "human_catchable", "note"]].tolist() == \
        ["S001", "t", "HAL", "yes", "sign is fake"]
    assert any("S002" in s.value for s in at.subheader)  # advanced to next trial

    # fresh session resumes at the first unannotated trial
    at2 = AppTest.from_file(str(SRC), default_timeout=30).run()
    assert any("S002" in s.value for s in at2.subheader)
