"""Metrics and error analysis for the AI outputs. Prints markdown.

python evaluate.py [data/study_dataset.csv] [data/annotations.csv]
If the annotations file exists, its final labels are also written into the dataset CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["United States", "Germany", "Japan", "South Korea", "India"]
LABELS = ["error_category", "rationale_faithful_top1", "human_catchable"]
csv = Path(sys.argv[1] if len(sys.argv) > 1 else "data/study_dataset.csv")
ann_csv = Path(sys.argv[2] if len(sys.argv) > 2 else "data/annotations.csv")
df = pd.read_csv(csv, dtype={"trial_id": str}, keep_default_na=False)
df[["ai_correct_top1", "ai_correct_top3"]] = df[["ai_correct_top1", "ai_correct_top3"]].astype(str) == "True"
n, right, p1 = len(df), df.ai_correct_top1.to_numpy(float), df.ai_p1.to_numpy(float)


def table(header, rows):
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in [header, ["---"] * len(header), *rows])


def ci(values):  # 95% bootstrap CI over trials, 1000 resamples
    means = np.asarray(values, float)[np.random.default_rng(42).integers(0, n, (1000, n))].mean(axis=1)
    return f"{np.quantile(means, 0.025):.3f}–{np.quantile(means, 0.975):.3f}"


b = np.minimum((p1 * 10).astype(int), 9)  # 10 equal-width bins on top-1 confidence, 1.0 in the last
ece = sum((b == i).sum() / n * abs(right[b == i].mean() - p1[b == i].mean()) for i in range(10) if (b == i).any())
metrics = [("Top-1 accuracy", right), ("Top-3 accuracy", df.ai_correct_top3), ("MRR", 1 / df.gt_rank)]
print(f"# {csv} (n = {n})\n")
print(table(["Metric", "Value", "95% CI"], [(m, f"{np.mean(v):.3f}", ci(v)) for m, v in metrics]
            + [("ECE (10 bins)", f"{ece:.4f}", ""), ("Chance", f"{1 / len(COUNTRIES):.3f}", "")]))

tercile = pd.cut(df.ai_p1.rank(method="max"), [0, n / 3, 2 * n / 3, n], labels=["low", "mid", "high"])
print("\n" + table(["Confidence tercile", "n", "Top-1 confidence", "Top-1 accuracy"],
                   [(t, len(g), f"{g.ai_p1.min():.2f}–{g.ai_p1.max():.2f} (mean {g.ai_p1.mean():.3f})",
                     f"{g.ai_correct_top1.mean():.3f}") for t, g in df.groupby(tercile, observed=True)]))
by = df.groupby("country")[["ai_correct_top1", "ai_correct_top3"]].agg(["size", "mean"])
print("\n" + table(["Country", "n", "Top-1", "Top-3"], [(c, by.loc[c, ("ai_correct_top1", "size")],
      f"{by.loc[c, ('ai_correct_top1', 'mean')]:.3f}", f"{by.loc[c, ('ai_correct_top3', 'mean')]:.3f}") for c in COUNTRIES]))
cm = pd.crosstab(df.country, df.ai_top1).reindex(index=COUNTRIES, columns=COUNTRIES, fill_value=0)
print("\n" + table(["Truth \\ AI top-1", *COUNTRIES], [(c, *cm.loc[c]) for c in COUNTRIES]))

if ann_csv.exists():
    ann = pd.read_csv(ann_csv, dtype=str, keep_default_na=False)
    df = df.drop(columns=LABELS, errors="ignore").merge(ann[["trial_id", *LABELS]], on="trial_id", how="left")
    df[LABELS] = df[LABELS].fillna("")
    df.to_csv(csv, index=False)
    err = df[~df.ai_correct_top1]
    groups = [(c, err[err.error_category == c]) for c in ("HAL", "TXT", "NBR", "KNW", "CUE")]
    groups.append(("*OVR (secondary: confidence >= 0.6 and wrong)*", err[err.ai_p1 >= 0.6]))
    print(f"\n## Error distribution ({len(err)} top-1 errors)\n")
    print(table(["Category", "# errors", "% of errors", "Human-catchable (yes / maybe / no)", "Example trial IDs"],
                [(c, len(g), f"{len(g) / len(err):.1%}",
                  " / ".join(str((g.human_catchable == k).sum()) for k in ("yes", "maybe", "no")),
                  ", ".join(sorted(g.trial_id)[:3])) for c, g in groups]))
    print(f"\nHAL + TXT: {err.error_category.isin(['HAL', 'TXT']).mean():.1%} of errors. Human-catchable: yes "
          f"{(err.human_catchable == 'yes').mean():.1%}, yes or maybe {err.human_catchable.isin(['yes', 'maybe']).mean():.1%}.")
    rated = df[df.rationale_faithful_top1 != ""]
    faith = ("faithful", "partially", "hallucinated", "none")
    print("\n" + table(["Top-1 rationale", "n", *faith], [
        (name, len(g), *(f"{(g.rationale_faithful_top1 == f).mean():.1%}" for f in faith))
        for name, g in [("AI wrong", rated[~rated.ai_correct_top1]), ("AI right", rated[rated.ai_correct_top1])]]))

    def kappa(col, rows):  # Cohen's kappa between the two raters, before adjudication
        a, b = ann.loc[rows, f"sonnet_{col}"].to_numpy(), ann.loc[rows, f"haiku_{col}"].to_numpy()
        pe = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
        return ((a == b).mean() - pe) / (1 - pe)

    wrong = ann.trial_id.isin(err.trial_id)
    print(f"\nInter-rater agreement (Sonnet vs Haiku): error_category κ = {kappa('error_category', wrong):.2f}, "
          f"human_catchable κ = {kappa('human_catchable', wrong):.2f}, "
          f"rationale_faithful_top1 κ = {kappa('rationale_faithful_top1', ann.index):.2f}")
