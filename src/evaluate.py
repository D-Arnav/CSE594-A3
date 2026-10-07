"""Evaluate AI predictions: accuracy, calibration, breakdowns, error table (SPEC 7, 8.3).

python src/evaluate.py --pred outputs/study_dataset.csv --ann outputs/annotations.csv
python src/evaluate.py --pred outputs/study_dataset.csv          # evaluation-only path
"""
import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLUSTER_OF, CLUSTERS, ROOT, countries, load_config  # noqa: E402

CATEGORIES = ["HAL", "TXT", "NBR", "KNW", "CUE"]
ANN_COLS = ["error_category", "rationale_faithful_top1", "human_catchable"]
FAITH = ["faithful", "partially", "hallucinated", "none"]
CATCH = ["yes", "maybe", "no"]
OVR_P = 0.6
NO_ANN = "annotations not provided"
# ponytail: phrase heuristic for "rationale contradicts ranking" (SPEC 15); will miss paraphrases
CONTRA = re.compile(r"argues? against|not consistent|inconsistent with|unlikely|"
                    r"does not support|doesn't support|rules? out", re.I)
# Okabe-Ito (colorblind-safe), one per cluster; grey for anything else
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
CLUSTER_COLOR = {k: OKABE_ITO[i] if i < len(OKABE_ITO) else "#999999" for i, k in enumerate(CLUSTERS)}


# ---------- loading ----------

def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_pred(path):
    df = pd.read_csv(path)
    for c in ANN_COLS + ["ai_rationale1", "ai_rationale2", "ai_rationale3"]:
        df[c] = df[c].fillna("").astype(str).str.strip() if c in df else ""
    df["error_category"] = df["error_category"].str.upper()
    df["ai_p1"] = df["ai_p1"].astype(float)
    df["has_text"] = _truthy(df["has_text"])
    df["correct1"] = df["ai_top1"] == df["country"]
    df["correct3"] = df[["ai_top1", "ai_top2", "ai_top3"]].eq(df["country"], axis=0).any(axis=1)
    df["gt_rank"] = df["gt_rank"].astype(float)
    df["ovr"] = (df["ai_p1"] >= OVR_P) & ~df["correct1"]
    return df


def load_annotations(path):
    """Annotation CSV -> one row per trial_id (latest timestamp wins, file order breaks ties)."""
    ann = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ANN_COLS + ["trial_id"]:
        ann[c] = ann[c].str.strip() if c in ann else ""
    ann["error_category"] = ann["error_category"].str.upper()
    ann["_ts"] = pd.to_datetime(ann.get("timestamp", ""), errors="coerce", utc=True, format="ISO8601")
    ann = ann.sort_values("_ts", kind="stable", na_position="first")
    return ann.drop_duplicates("trial_id", keep="last").drop(columns="_ts").reset_index(drop=True)


def merge_annotations(df, ann):
    """Overwrite annotation columns of df for trials present in ann."""
    df = df.copy()
    vals = ann.set_index("trial_id")
    for c in ANN_COLS:
        df[c] = df["trial_id"].map(vals[c]).fillna(df[c])
    return df


def write_back(pred_path, df):
    """Write annotation columns into the pred CSV in place, keeping every other cell as-is."""
    raw = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    vals = df.set_index("trial_id")
    for c in ANN_COLS:
        if c not in raw:
            raw[c] = ""
        raw[c] = raw["trial_id"].map(vals[c]).fillna(raw[c])
    raw.to_csv(pred_path, index=False)


def has_annotations(df):
    return bool((df[ANN_COLS] != "").any().any())


# ---------- metrics ----------

def _f(x):
    """float, with NaN -> None so metrics.json stays valid JSON."""
    x = float(x)
    return None if math.isnan(x) else x


def bootstrap_ci(values, n_boot, seed, alpha=0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def ece(conf, correct, n_bins):
    """Guo et al. 2017: sum_b n_b/N |acc_b - conf_b|, equal-width bins [lo, hi) (last bin closed)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        acc = float(correct[m].mean()) if n else None
        cf = float(conf[m].mean()) if n else None
        if n:
            total += n / len(conf) * abs(acc - cf)
        bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n, "acc": acc, "conf": cf})
    return total, bins


def tercile_table(df):
    n = len(df)  # rank "max" keeps tied ai_p1 in one tercile
    t = pd.cut(df["ai_p1"].rank(method="max"), [0, n / 3, 2 * n / 3, n], labels=["low", "mid", "high"])
    rows = []
    for name, g in df.groupby(t, observed=True):
        rows.append({"tercile": name, "n": len(g), "p1_min": float(g.ai_p1.min()),
                     "p1_max": float(g.ai_p1.max()), "mean_p1": float(g.ai_p1.mean()),
                     "acc": float(g.correct1.mean())})
    return rows


def group_accuracy(df, col):
    return {str(k): {"n": len(g), "top1": float(g.correct1.mean()), "top3": float(g.correct3.mean())}
            for k, g in df.groupby(col)}


def same_cluster(a, b):
    return a.map(CLUSTER_OF).fillna("?a") == b.map(CLUSTER_OF).fillna("?b")


def within_cluster_error_rate(df):
    err = df[~df.correct1]
    return float(same_cluster(err.ai_top1, err.country).mean()) if len(err) else None


def cluster_order(labels):
    order = list(CLUSTERS)
    return sorted(labels, key=lambda c: order.index(CLUSTER_OF[c]) if c in CLUSTER_OF else len(order))


def confusion(df, labels):
    pos = {c: i for i, c in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(df.country, df.ai_top1):
        if t in pos and p in pos:
            cm[pos[t], pos[p]] += 1
    return cm


def contradiction_rate(df):
    hit = df["ai_rationale1"].str.contains(CONTRA)
    return {"rate": float(hit.mean()), "n": int(hit.sum()), "trial_ids": df.trial_id[hit].tolist()}


def length_normalized(raw_path, df):
    """Top-1/top-3 when ranking by sum logprob vs logprob / n_answer_tokens (SPEC 15)."""
    rows = {}
    for line in Path(raw_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["trial_id"]] = r  # later lines (resumes) win
    truth = dict(zip(df.trial_id, df.country))
    raw1 = raw3 = norm1 = norm3 = agree = n = 0
    for tid, r in rows.items():
        if tid not in truth or not r.get("n_answer_tokens"):
            continue
        lp, nt = r["logprobs"], r["n_answer_tokens"]
        by_raw = sorted(lp, key=lambda c: lp[c], reverse=True)
        by_norm = sorted(lp, key=lambda c: lp[c] / max(nt.get(c, 1), 1), reverse=True)
        t = truth[tid]
        n += 1
        raw1 += by_raw[0] == t
        raw3 += t in by_raw[:3]
        norm1 += by_norm[0] == t
        norm3 += t in by_norm[:3]
        agree += by_raw[0] == by_norm[0]
    if not n:
        return {"n": 0, "note": "no rows with n_answer_tokens"}
    return {"n": n, "raw_top1": raw1 / n, "raw_top3": raw3 / n, "norm_top1": norm1 / n,
            "norm_top3": norm3 / n, "top1_agreement": agree / n}


# ---------- annotation analysis ----------

def catch_counts(g):
    return [int((g.human_catchable == c).sum()) for c in CATCH]


def error_table(df):
    """SPEC 8.3 rows + summary numbers. Percentages are over all top-1 errors."""
    err = df[~df.correct1]
    n_err = len(err)
    rows = []
    for cat in CATEGORIES + ["OVR"]:
        g = err[err.ovr] if cat == "OVR" else err[err.error_category == cat]
        rows.append({"category": cat, "n": len(g), "pct": len(g) / n_err if n_err else 0.0,
                     "catchable": catch_counts(g), "examples": sorted(g.trial_id)[:3]})
    yes, maybe, _ = catch_counts(err)
    summary = {
        "n_errors": n_err,
        "n_unannotated_errors": int((err.error_category == "").sum()),
        "hal_txt_share": _f(err.error_category.isin(["HAL", "TXT"]).mean()) if n_err else None,
        "catchable_yes_share": yes / n_err if n_err else None,
        "catchable_yes_maybe_share": (yes + maybe) / n_err if n_err else None,
        "n_ovr": int(err.ovr.sum()),
    }
    return rows, summary


def error_table_md(rows, summary):
    lines = ["| Category | # errors | % of errors | Human-catchable (yes / maybe / no) | Example trial IDs |",
             "|---|---|---|---|---|"]
    for r in rows:
        name = "*OVR (secondary)*" if r["category"] == "OVR" else r["category"]
        lines.append(f"| {name} | {r['n']} | {100 * r['pct']:.1f}% | "
                     f"{' / '.join(map(str, r['catchable']))} | {', '.join(r['examples'])} |")
    s, pct = summary, lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines += ["",
              f"Percentages are over all {s['n_errors']} top-1 errors. "
              f"OVR = top-1 confidence >= {OVR_P} and wrong (derived, can co-occur).",
              f"- HAL + TXT share of errors: {pct(s['hal_txt_share'])}",
              f"- Human-catchable share of errors: yes {pct(s['catchable_yes_share'])}, "
              f"yes+maybe {pct(s['catchable_yes_maybe_share'])} (SPEC 0.3 target >= 40%)"]
    if s["n_unannotated_errors"]:
        lines.append(f"- WARNING: {s['n_unannotated_errors']} of {s['n_errors']} top-1 errors have no error_category.")
    return "\n".join(lines) + "\n"


def faithfulness_rates(df):
    out = {}
    for name, g in [("errors", df[~df.correct1]), ("correct", df[df.correct1])]:
        g = g[g.rationale_faithful_top1 != ""]
        out[name] = {"n": len(g), **{f: (float((g.rationale_faithful_top1 == f).mean()) if len(g) else None)
                                      for f in FAITH}}
    return out


def kappa(ann1, ann2, df):
    """Cohen's kappa on overlapping trials (error_category / human_catchable on top-1 errors only)."""
    from sklearn.metrics import cohen_kappa_score
    m = ann1.merge(ann2, on="trial_id", suffixes=("_1", "_2"))
    wrong = set(df.trial_id[~df.correct1])
    out = {"n_overlap": len(m)}
    for col in ANN_COLS:
        s = m if col == "rationale_faithful_top1" else m[m.trial_id.isin(wrong)]
        s = s[(s[col + "_1"] != "") & (s[col + "_2"] != "")]
        k = None
        if len(s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k = _f(cohen_kappa_score(s[col + "_1"], s[col + "_2"]))
        out[col] = {"n": len(s), "kappa": k,
                    "agreement": float((s[col + "_1"] == s[col + "_2"]).mean()) if len(s) else None}
    return out


# ---------- figures ----------

def plot_reliability(bins, ece_val, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    w = bins[0]["hi"] - bins[0]["lo"]
    for b in bins:
        if b["n"]:
            ax.bar(b["lo"] + w / 2, b["acc"], width=w * 0.95, color="#0072B2", edgecolor="black")
            ax.text(b["lo"] + w / 2, b["acc"] + 0.01, f"n={b['n']}", ha="center", va="bottom", fontsize=8)
    ax.plot([0, 1], [0, 1], "--", color="#D55E00", label="perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="Top-1 confidence (ai_p1)", ylabel="Top-1 accuracy",
           title=f"Reliability diagram (ECE = {ece_val:.3f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _cluster_spans(labels):
    spans, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or CLUSTER_OF.get(labels[i]) != CLUSTER_OF.get(labels[start]):
            spans.append((start, i))
            start = i
    return spans


def plot_confusion(cm, labels, path):
    side = 3 + 0.45 * len(labels)  # 20 labels -> 12 in, 5 -> ~5 in
    fig, ax = plt.subplots(figsize=(side + 1, side))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# trials")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    for a, b in _cluster_spans(labels):
        for edge in (a - 0.5, b - 0.5):
            ax.axhline(edge, color="grey", lw=0.8)
            ax.axvline(edge, color="grey", lw=0.8)
        # within-cluster block: thick orange outline
        ax.add_patch(plt.Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec="#E69F00", lw=2.5))
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set(xlabel="AI top-1 prediction", ylabel="Ground truth",
           title="Confusion matrix (ordered by cluster; orange = within-cluster)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_country(per_country, labels, overall, chance, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 5))
    acc = [per_country.get(c, {}).get("top1", 0.0) for c in labels]
    ax.bar(labels, acc, color=[CLUSTER_COLOR.get(CLUSTER_OF.get(c), "#999999") for c in labels],
           edgecolor="black")
    ax.axhline(overall, color="black", ls="--", lw=1, label=f"overall {overall:.2f}")
    ax.axhline(chance, color="grey", ls=":", lw=1, label=f"chance {chance:.2f}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLUSTER_COLOR[k]) for k in CLUSTERS
               if any(CLUSTER_OF.get(c) == k for c in labels)]
    names = [k for k in CLUSTERS if any(CLUSTER_OF.get(c) == k for c in labels)]
    lines, line_names = ax.get_legend_handles_labels()
    ax.legend(handles + lines, names + line_names, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Per-country top-1 accuracy (colored by cluster)")
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- report ----------

def md_table(headers, rows):
    return "\n" + "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]) + "\n"


def summary_md(m, err_md):
    p = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    ci = lambda k, f=p: f"{f(m[k]['value'])} [{f(m[k]['ci'][0])}, {f(m[k]['ci'][1])}]"  # noqa: E731
    out = [f"# Evaluation report\n\nPredictions: `{m['pred']}` · model `{m['model_id']}` @ `{m['revision']}` · n = {m['n']} · "
           f"{m['bootstrap']} bootstrap resamples (seed {m['seed']})\n\n",
           "## Primary metrics (95% bootstrap CI)\n",
           md_table(["Metric", "Value"], [
               ["Top-1 accuracy", ci("top1")], ["Top-3 accuracy", ci("top3")],
               ["Mean reciprocal rank", ci("mrr", lambda x: f"{x:.3f}")], ["Same-cluster partial credit", ci("partial_credit")],
               ["Chance baseline", p(m["chance"])]]),
           f"\n## Calibration\n\nECE ({m['ece_bins']} bins) = {m['ece']:.4f}. See `reliability.png`.\n",
           md_table(["Tercile", "n", "ai_p1 range", "mean ai_p1", "Top-1 acc"],
                    [[t["tercile"], t["n"], f"{t['p1_min']:.3f}–{t['p1_max']:.3f}", f"{t['mean_p1']:.3f}",
                      p(t["acc"])] for t in m["terciles"]]),
           "\n## Accuracy by has_text\n",
           md_table(["has_text", "n", "Top-1", "Top-3"],
                    [[k, v["n"], p(v["top1"]), p(v["top3"])] for k, v in m["by_has_text"].items()]),
           "\n## Per-country accuracy\n",
           md_table(["Country", "Cluster", "n", "Top-1", "Top-3"],
                    [[c, CLUSTER_OF.get(c, "?"), v["n"], p(v["top1"]), p(v["top3"])]
                     for c, v in m["per_country"].items()]),
           f"\nWithin-cluster error rate (share of top-1 errors where prediction and truth share a cluster): "
           f"{p(m['within_cluster_error_rate'])} of {m['n_errors']} errors. See `confusion_matrix.png`.\n",
           f"\nRationale-contradicts-ranking heuristic (top-1 rationale matches e.g. \"argues against\", "
           f"\"unlikely\"): {p(m['rationale_contradiction']['rate'])} "
           f"({m['rationale_contradiction']['n']} trials).\n"]
    ln = m.get("length_normalized")
    if ln and ln.get("n"):
        out += ["\n## Length-normalized ranking (SPEC 15)\n",
                md_table(["Ranking", "Top-1", "Top-3"],
                         [["sum logprob", p(ln["raw_top1"]), p(ln["raw_top3"])],
                          ["logprob / n_answer_tokens", p(ln["norm_top1"]), p(ln["norm_top3"])]]),
                f"\nTop-1 agreement between the two rankings: {p(ln['top1_agreement'])} (n = {ln['n']}).\n"]
    out.append("\n## Error analysis\n")
    if m["annotations"] == NO_ANN:
        out.append(NO_ANN + ".\n")
    else:
        a = m["annotations"]
        out += [f"Source: {a['source']}\n\n", err_md, "\n### Rationale faithfulness (top-1)\n",
                md_table(["Trials", "n"] + FAITH,
                         [[k, v["n"]] + [p(v[f]) for f in FAITH] for k, v in a["faithfulness"].items()])]
        if "kappa" in a:
            k = a["kappa"]
            out += [f"\n### Inter-rater agreement ({k['n_overlap']} overlapping trials)\n",
                    md_table(["Field", "n", "Cohen's kappa", "Raw agreement"],
                             [[c, k[c]["n"], "n/a" if k[c]["kappa"] is None else f"{k[c]['kappa']:.3f}",
                               p(k[c]["agreement"])] for c in ANN_COLS])]
    out += ["\n## Success criteria (SPEC 0.3)\n",
            md_table(["Criterion", "Target", "Value", "Pass"],
                     [[k, v["target"], v["value"], {True: "PASS", False: "FAIL", None: "n/a"}[v["pass"]]]
                      for k, v in m["success_criteria"].items()])]
    return "".join(out)


def evaluate(pred, out, cfg, ann=None, ann2=None, raw=None, merge=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = cluster_order(countries(cfg))
    n_boot = int(cfg["eval"].get("bootstrap", 1000))
    n_bins = int(cfg["eval"].get("ece_bins", 10))
    seed = int(cfg.get("seed", 42))

    df = load_pred(pred)
    ann_df = None
    if ann:
        ann_df = load_annotations(ann)
        df = merge_annotations(df, ann_df)
        if merge:
            write_back(pred, df)

    n = len(df)
    m = {"pred": str(pred), "n": n, "seed": seed, "bootstrap": n_boot, "chance": 1 / len(labels),
         "n_labels": len(labels), "ece_bins": n_bins,
         "model_id": "|".join(df.model_id.astype(str).unique()),
         "revision": "|".join(df.revision.astype(str).unique())}
    df["partial"] = same_cluster(df.ai_top1, df.country)
    for key, vals in [("top1", df.correct1), ("top3", df.correct3), ("mrr", 1 / df.gt_rank),
                      ("partial_credit", df.partial)]:
        m[key] = {"value": float(vals.mean()), "ci": bootstrap_ci(vals, n_boot, seed)}
    m["ece"], m["reliability_bins"] = ece(df.ai_p1, df.correct1, n_bins)
    m["terciles"] = tercile_table(df)
    m["by_has_text"] = group_accuracy(df, "has_text")
    per = group_accuracy(df, "country")
    m["per_country"] = {c: per[c] for c in labels if c in per}
    m["n_errors"] = int((~df.correct1).sum())
    m["within_cluster_error_rate"] = within_cluster_error_rate(df)
    m["rationale_contradiction"] = contradiction_rate(df)
    if raw:
        m["length_normalized"] = length_normalized(raw, df)

    err_md = ""
    if has_annotations(df):
        rows, s = error_table(df)
        err_md = error_table_md(rows, s)
        m["annotations"] = {"source": str(ann) if ann else f"annotation columns already in {pred}",
                            "error_table": rows, **s, "faithfulness": faithfulness_rates(df)}
        if ann2 and ann_df is not None:
            m["annotations"]["kappa"] = kappa(ann_df, load_annotations(ann2), df)
    else:
        err_md = NO_ANN + "\n"
        m["annotations"] = NO_ANN
    (out / "error_table.md").write_text(err_md)

    counts = df.country.value_counts()
    a = m["annotations"] if isinstance(m["annotations"], dict) else {}
    yes = a.get("catchable_yes_share")
    m["success_criteria"] = {
        "study_set_size": {"target": "200 trials, 10 per country", "value": f"{n} trials, "
                           f"{counts.min()}-{counts.max()} per country over {len(counts)} countries",
                           "pass": bool(n == 200 and len(counts) == len(labels) and (counts == 10).all())},
        "top1_accuracy": {"target": "0.45-0.75", "value": round(m["top1"]["value"], 4),
                          "pass": bool(0.45 <= m["top1"]["value"] <= 0.75)},
        "top3_accuracy": {"target": ">= 0.75", "value": round(m["top3"]["value"], 4),
                          "pass": bool(m["top3"]["value"] >= 0.75)},
        "human_catchable_share": {
            "target": ">= 0.40 of top-1 errors with human_catchable=yes",
            "value": None if yes is None else round(yes, 4),
            "value_yes_or_maybe": a.get("catchable_yes_maybe_share"),
            "pass": None if yes is None else bool(yes >= 0.40)},
    }

    plot_reliability(m["reliability_bins"], m["ece"], out / "reliability.png")
    plot_confusion(confusion(df, labels), labels, out / "confusion_matrix.png")
    plot_per_country(m["per_country"], labels, m["top1"]["value"], m["chance"], out / "per_country_accuracy.png")
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    (out / "summary.md").write_text(summary_md(m, err_md))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ann")
    ap.add_argument("--ann2")
    ap.add_argument("--raw")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-merge", action="store_true", help="do not write annotations back into --pred")
    args = ap.parse_args()
    m = evaluate(args.pred, args.out, load_config(args.config), ann=args.ann, ann2=args.ann2,
                 raw=args.raw, merge=not args.no_merge)
    print(f"n={m['n']} top1={m['top1']['value']:.3f} {m['top1']['ci']} top3={m['top3']['value']:.3f} "
          f"mrr={m['mrr']['value']:.3f} ece={m['ece']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
