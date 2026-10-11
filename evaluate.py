"""The two tables in the report: AI metrics and error analysis.   python evaluate.py"""
import numpy as np
import pandas as pd

df = pd.read_csv("data/study_dataset.csv", dtype={"trial_id": str}, keep_default_na=False)
ann = pd.read_csv("data/annotations.csv", dtype={"trial_id": str}, keep_default_na=False)
right, p1 = (df.ai_correct_top1.astype(str) == "True").to_numpy(float), df.ai_p1.to_numpy(float)

# Expected calibration error: 10 equal-width bins on top-1 confidence (1.0 goes in the last bin)
b = np.minimum((p1 * 10).astype(int), 9)
ece = sum((b == i).mean() * abs(right[b == i].mean() - p1[b == i].mean()) for i in range(10) if (b == i).any())

print("| Metric | Value |\n|---|---|")
print(f"| Top-1 accuracy | {right.mean():.1%} |")
print(f"| Top-3 accuracy | {(df.ai_correct_top3.astype(str) == 'True').mean():.1%} |")
print(f"| Mean reciprocal rank | {(1 / df.gt_rank).mean():.3f} |")
print(f"| Expected calibration error | {ece:.3f} |\n")

err = ann[ann.trial_id.isin(df.trial_id[right == 0])]  # labels for the AI's top-1 errors
print("| Category | # errors | % of errors | Human-catchable (yes / maybe / no) | Examples |\n|---|---|---|---|---|")
for cat in ("HAL", "TXT", "KNW", "CUE"):
    g = err[err.error_category == cat]
    catch = " / ".join(str((g.human_catchable == c).sum()) for c in ("yes", "maybe", "no"))
    print(f"| {cat} | {len(g)} | {len(g) / len(err):.1%} | {catch} | {', '.join(sorted(g.trial_id)[:3])} |")
