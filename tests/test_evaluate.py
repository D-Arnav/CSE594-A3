import json
import subprocess
import sys

import pandas as pd
import pytest

import evaluate as ev
from common import CLUSTER_OF, ROOT, load_config

CFG = load_config()
# Synthetic data uses its own 20-country label space (independent of config.yaml's study countries).
CFG["data"]["countries"] = ["United States", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
                            "United Kingdom", "Ireland", "France", "Spain", "Italy", "Germany",
                            "Russia", "Poland", "Japan", "South Korea", "India", "Thailand",
                            "South Africa", "Australia"]
CFG["eval"]["bootstrap"] = 200
LABELS = list(CFG["data"]["countries"])
COLS = ["trial_id", "image_path", "source_id", "country", "cluster", "has_text",
        "ai_top1", "ai_top2", "ai_top3", "ai_p1", "ai_p2", "ai_p3",
        "ai_rationale1", "ai_rationale2", "ai_rationale3", "ai_correct_top1", "ai_correct_top3",
        "gt_rank", "error_category", "rationale_faithful_top1", "human_catchable",
        "model_id", "revision", "prompt_version", "temperature", "max_pixels"]


@pytest.fixture(autouse=True)
def fixed_labels(monkeypatch):
    # ignore any data/countries.txt so the synthetic label space is known
    monkeypatch.setattr(ev, "countries", lambda cfg: list(cfg["data"]["countries"]))


def tid(i):
    return f"S{i + 1:03d}"


def neighbor(c):
    return next(x for x in LABELS if x != c and CLUSTER_OF[x] == CLUSTER_OF[c])


def stranger(c):
    return next(x for x in LABELS if CLUSTER_OF[x] != CLUSTER_OF[c])


def others(c, *exclude):
    return [x for x in LABELS if x != c and x not in exclude]


def make_pred():
    """40 trials, 2 per country. Row 2k correct (p=.9). Row 2k+1 wrong:
    k<10 same-cluster pred with truth at rank 2 (p=.5); k>=10 other-cluster pred, truth rank 5
    (p=.7 for k=10..12 -> OVR, else .3). has_text: correct rows + wrong rows with k<5."""
    rows = []
    for i in range(40):
        k, c = i // 2, LABELS[i // 2]
        if i % 2 == 0:
            top, p1, rank, text = [c] + others(c)[:2], 0.9, 1, True
        elif k < 10:
            top, p1, rank, text = [neighbor(c), c, others(c, neighbor(c))[0]], 0.5, 2, k < 5
        else:
            s = stranger(c)
            top, p1, rank, text = [s] + others(c, s)[:2], (0.7 if k <= 12 else 0.3), 5, False
        rows.append({"trial_id": tid(i), "image_path": f"data/study/{tid(i)}.jpg", "source_id": 1000 + i,
                     "country": c, "cluster": CLUSTER_OF[c], "has_text": text,
                     "ai_top1": top[0], "ai_top2": top[1], "ai_top3": top[2],
                     "ai_p1": p1, "ai_p2": 0.05, "ai_p3": 0.01,
                     "ai_rationale1": "The flora argues against it." if i == 1 else "Road markings, signs.",
                     "ai_rationale2": "r2", "ai_rationale3": "r3",
                     "ai_correct_top1": top[0] == c, "ai_correct_top3": c in top, "gt_rank": rank,
                     "error_category": "", "rationale_faithful_top1": "", "human_catchable": "",
                     "model_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": "abc",
                     "prompt_version": "rank=v1;explain=v1", "temperature": 1.3, "max_pixels": 262144})
    return pd.DataFrame(rows, columns=COLS)


def make_ann():
    """Errors are rows 2k+1. k0-3 HAL/yes, k4-5 TXT/yes, k6-9 NBR/maybe, k10-11 KNW/maybe,
    k12-13 KNW/no, k14-17 CUE/no, k18-19 unannotated. 5 correct trials: 4 faithful, 1 none."""
    rows = []
    for k in range(18):
        cat = ["HAL"] * 4 + ["TXT"] * 2 + ["NBR"] * 4 + ["KNW"] * 4 + ["CUE"] * 4
        catch = ["yes"] * 6 + ["maybe"] * 6 + ["no"] * 6
        rows.append({"trial_id": tid(2 * k + 1), "rater": "r1", "error_category": cat[k],
                     "rationale_faithful_top1": "hallucinated" if cat[k] == "HAL" else "partially",
                     "human_catchable": catch[k], "note": "", "timestamp": f"2026-09-01T10:{k:02d}:00"})
    for j, f in enumerate(["faithful"] * 4 + ["none"]):
        rows.append({"trial_id": tid(2 * j), "rater": "r1", "error_category": "",
                     "rationale_faithful_top1": f, "human_catchable": "", "note": "",
                     "timestamp": "2026-09-01T11:00:00"})
    # stale duplicate (older timestamp, later in file) must lose
    rows.append({"trial_id": tid(1), "rater": "r1", "error_category": "CUE", "rationale_faithful_top1": "none",
                 "human_catchable": "no", "note": "old", "timestamp": "2026-08-01T00:00:00"})
    return pd.DataFrame(rows)


@pytest.fixture
def files(tmp_path):
    pred, ann = tmp_path / "study_dataset.csv", tmp_path / "annotations.csv"
    make_pred().to_csv(pred, index=False)
    make_ann().to_csv(ann, index=False)
    return pred, ann, tmp_path / "report"


def test_ece_tiny_hand_computed():
    # bin 9: conf .95, acc .5 -> .45*2/4 ; bin 5: conf .55, acc 1 -> .45*2/4
    val, bins = ev.ece([0.95, 0.95, 0.55, 0.55], [1, 0, 1, 1], 10)
    assert val == pytest.approx(0.45)
    assert [b["n"] for b in bins] == [0, 0, 0, 0, 0, 2, 0, 0, 0, 2]
    assert ev.ece([1.0, 0.0], [1, 0], 10)[0] == pytest.approx(0.0)  # edges land in first/last bin


def test_tercile_ties_stay_together():
    df = pd.DataFrame({"ai_p1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] + [1.0] * 6, "correct1": [0, 1] * 6})
    for seed in range(5):
        t = {r["tercile"]: r for r in ev.tercile_table(df.sample(frac=1, random_state=seed))}
        assert t["mid"]["p1_max"] < 1.0 and t["high"]["p1_min"] == 1.0 and t["high"]["n"] == 6


def test_bootstrap_ci_contains_point_and_is_seeded():
    vals = [1] * 30 + [0] * 70
    lo, hi = ev.bootstrap_ci(vals, 500, seed=1)
    assert lo < 0.3 < hi
    assert ev.bootstrap_ci(vals, 500, seed=1) == [lo, hi]


def test_core_metrics_eval_only(files):
    pred, _, out = files
    m = ev.evaluate(pred, out, CFG)
    assert m["n"] == 40
    assert m["top1"]["value"] == pytest.approx(0.5)
    assert m["top3"]["value"] == pytest.approx(0.75)
    assert m["mrr"]["value"] == pytest.approx((20 * 1 + 10 * 0.5 + 10 * 0.2) / 40)
    assert m["partial_credit"]["value"] == pytest.approx(0.75)
    assert m["chance"] == pytest.approx(0.05)
    assert m["within_cluster_error_rate"] == pytest.approx(0.5)
    # bins: .9 x20 acc1, .5 x10 acc0, .7 x3 acc0, .3 x7 acc0
    assert m["ece"] == pytest.approx(0.1 * 20 / 40 + 0.5 * 10 / 40 + 0.7 * 3 / 40 + 0.3 * 7 / 40)
    for k in ("top1", "top3", "mrr", "partial_credit"):
        lo, hi = m[k]["ci"]
        assert lo <= m[k]["value"] <= hi
    t = {r["tercile"]: r for r in m["terciles"]}
    assert sum(r["n"] for r in t.values()) == 40
    assert t["low"]["acc"] == 0.0 and t["high"]["acc"] == 1.0
    assert t["low"]["p1_max"] <= t["mid"]["p1_min"] <= t["high"]["p1_min"]
    assert (m["model_id"], m["revision"]) == ("Qwen/Qwen3-VL-2B-Instruct", "abc")
    assert "abc" in json.loads((out / "metrics.json").read_text())["revision"]
    assert m["by_has_text"]["True"] == {"n": 25, "top1": 0.8, "top3": 1.0}
    assert m["by_has_text"]["False"]["n"] == 15 and m["by_has_text"]["False"]["top1"] == 0.0
    assert m["per_country"][LABELS[0]]["top1"] == 0.5
    assert m["rationale_contradiction"]["trial_ids"] == ["S002"]
    # no annotations anywhere
    assert m["annotations"] == ev.NO_ANN
    assert "annotations not provided" in (out / "error_table.md").read_text()
    assert m["success_criteria"]["human_catchable_share"]["pass"] is None
    assert m["success_criteria"]["top1_accuracy"]["pass"] is True
    assert m["success_criteria"]["top3_accuracy"]["pass"] is True
    assert m["success_criteria"]["study_set_size"]["pass"] is False
    for f in ("metrics.json", "summary.md", "reliability.png", "confusion_matrix.png", "per_country_accuracy.png"):
        assert (out / f).stat().st_size > 0
    json.loads((out / "metrics.json").read_text())


def test_confusion_matrix_cluster_order():
    labels = ev.cluster_order(LABELS[::-1])
    assert [CLUSTER_OF[c] for c in labels] == sorted((CLUSTER_OF[c] for c in labels),
                                                     key=list(ev.CLUSTERS).index)
    spans = ev._cluster_spans(labels)
    assert spans[0] == (0, 3) and spans[-1][1] == 20 and len(spans) == 8


def test_confusion_counts(tmp_path):
    p = tmp_path / "p.csv"
    make_pred().to_csv(p, index=False)
    df = ev.load_pred(p)
    labels = ev.cluster_order(LABELS)
    cm = ev.confusion(df, labels)
    assert cm.sum() == 40 and cm.trace() == 20
    us, ca = labels.index("United States"), labels.index("Canada")
    assert cm[us, us] == 1 and cm[us, ca] == 1


def test_annotations_error_table_and_merge(files):
    pred, ann, out = files
    before = pd.read_csv(pred, dtype=str, keep_default_na=False)
    m = ev.evaluate(pred, out, CFG, ann=ann)
    a = m["annotations"]
    rows = {r["category"]: r for r in a["error_table"]}
    assert {c: rows[c]["n"] for c in rows} == {"HAL": 4, "TXT": 2, "NBR": 4, "KNW": 4, "CUE": 4, "OVR": 3}
    assert rows["HAL"]["pct"] == pytest.approx(0.2)
    assert rows["HAL"]["catchable"] == [4, 0, 0]
    assert rows["KNW"]["catchable"] == [0, 2, 2]
    assert rows["HAL"]["examples"] == ["S002", "S004", "S006"]  # stale CUE row for S002 lost
    # OVR derived: k=10,11,12 -> S022, S024, S026 ; catchable maybe, maybe, no
    assert rows["OVR"]["examples"] == ["S022", "S024", "S026"] and rows["OVR"]["catchable"] == [0, 2, 1]
    assert a["n_errors"] == 20 and a["n_unannotated_errors"] == 2
    assert a["hal_txt_share"] == pytest.approx(0.3)
    assert a["catchable_yes_share"] == pytest.approx(0.3)
    assert a["catchable_yes_maybe_share"] == pytest.approx(0.6)
    assert a["faithfulness"]["errors"]["n"] == 18
    assert a["faithfulness"]["errors"]["hallucinated"] == pytest.approx(4 / 18)
    assert a["faithfulness"]["correct"] == {"n": 5, "faithful": 0.8, "partially": 0.0,
                                            "hallucinated": 0.0, "none": 0.2}
    assert m["success_criteria"]["human_catchable_share"]["pass"] is False

    md = (out / "error_table.md").read_text()
    assert md.startswith("| Category | # errors | % of errors | Human-catchable (yes / maybe / no) "
                         "| Example trial IDs |\n|---|---|---|---|---|\n")
    assert "| HAL | 4 | 20.0% | 4 / 0 / 0 | S002, S004, S006 |" in md
    assert "| *OVR (secondary)* | 3 | 15.0% | 0 / 2 / 1 | S022, S024, S026 |" in md
    assert "WARNING: 2 of 20" in md

    # merge-back: same columns/order, non-annotation cells untouched, annotations written
    after = pd.read_csv(pred, dtype=str, keep_default_na=False)
    assert list(after.columns) == COLS
    keep = [c for c in COLS if c not in ev.ANN_COLS]
    pd.testing.assert_frame_equal(after[keep], before[keep])
    row = after.set_index("trial_id")
    assert row.loc["S002", "error_category"] == "HAL" and row.loc["S002", "human_catchable"] == "yes"
    assert row.loc["S001", "rationale_faithful_top1"] == "faithful"
    assert row.loc["S038", "error_category"] == ""  # unannotated error stays blank

    # GSI path: only --pred, annotations read back from the merged CSV
    m2 = ev.evaluate(pred, tmp := out.parent / "r2", CFG)
    assert m2["annotations"]["error_table"] == a["error_table"]
    assert (tmp / "error_table.md").read_text() == md


def test_no_merge_leaves_pred_untouched(files):
    pred, ann, out = files
    raw = pred.read_bytes()
    ev.evaluate(pred, out, CFG, ann=ann, merge=False)
    assert pred.read_bytes() == raw


def test_kappa(files):
    pred, ann, out = files
    a1 = pd.read_csv(ann)
    # rater 2 re-labels the first 10 errors (S002..S020): same except S020 NBR -> KNW
    a2 = a1[a1.trial_id.isin([tid(2 * k + 1) for k in range(10)]) & (a1.note != "old")].copy()
    a2.loc[a2.trial_id == "S020", "error_category"] = "KNW"
    p2 = out.parent / "ann2.csv"
    a2.to_csv(p2, index=False)
    k = ev.evaluate(pred, out, CFG, ann=ann, ann2=p2)["annotations"]["kappa"]
    assert k["n_overlap"] == 10
    # po = .9 ; pe = .4*.4 + .2*.2 + .4*.3 + 0*.1 = .32
    assert k["error_category"]["kappa"] == pytest.approx((0.9 - 0.32) / (1 - 0.32))
    assert k["error_category"]["agreement"] == pytest.approx(0.9)
    assert k["human_catchable"]["kappa"] == pytest.approx(1.0)


def test_length_normalized(files, tmp_path):
    pred, _, out = files
    c0, c1 = LABELS[0], LABELS[1]
    lines = [  # S001 truth c0: raw ranks c0 first, normalized ranks c1 first
        {"trial_id": "S001", "logprobs": {c0: -9.0, c1: -1.0}, "n_answer_tokens": {c0: 1, c1: 1}},  # stale
        {"trial_id": "S001", "logprobs": {c0: -2.0, c1: -3.0}, "n_answer_tokens": {c0: 1, c1: 4}},
        {"trial_id": "S002", "logprobs": {c0: -1.0, c1: -3.0}, "n_answer_tokens": {c0: 2, c1: 2}},
    ]
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    ln = ev.evaluate(pred, out, CFG, raw=raw)["length_normalized"]
    assert ln == {"n": 2, "raw_top1": 1.0, "raw_top3": 1.0, "norm_top1": 0.5, "norm_top3": 1.0,
                  "top1_agreement": 0.5}


def test_cli_eval_only(files):
    pred, _, out = files
    r = subprocess.run([sys.executable, str(ROOT / "src" / "evaluate.py"), "--pred", str(pred),
                        "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "top1=0.500" in r.stdout
    assert (out / "summary.md").exists()
