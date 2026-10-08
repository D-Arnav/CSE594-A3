# Evaluation report

Predictions: `outputs/study_dataset.csv` · model `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4` · n = 200 · 1000 bootstrap resamples (seed 42)

## Primary metrics (95% bootstrap CI)

| Metric | Value |
|---|---|
| Top-1 accuracy | 71.0% [64.5%, 77.0%] |
| Top-3 accuracy | 88.0% [83.0%, 92.5%] |
| Mean reciprocal rank | 0.814 [0.771, 0.850] |
| Same-cluster partial credit | 78.5% [73.0%, 84.0%] |
| Chance baseline | 20.0% |

## Calibration

ECE (10 bins) = 0.0937. See `reliability.png`.

| Tercile | n | ai_p1 range | mean ai_p1 | Top-1 acc |
|---|---|---|---|---|
| low | 64 | 0.290–0.590 | 0.464 | 29.7% |
| mid | 68 | 0.610–0.880 | 0.743 | 85.3% |
| high | 68 | 0.890–1.000 | 0.954 | 95.6% |

## Accuracy by has_text

| has_text | n | Top-1 | Top-3 |
|---|---|---|---|
| False | 179 | 69.8% | 86.6% |
| True | 21 | 81.0% | 100.0% |

## Per-country accuracy

| Country | Cluster | n | Top-1 | Top-3 |
|---|---|---|---|---|
| United States | North America | 40 | 100.0% | 100.0% |
| Germany | Western Europe | 40 | 15.0% | 42.5% |
| Japan | East Asia | 40 | 97.5% | 100.0% |
| South Korea | East Asia | 40 | 60.0% | 100.0% |
| India | South / Southeast Asia | 40 | 82.5% | 97.5% |

Within-cluster error rate (share of top-1 errors where prediction and truth share a cluster): 25.9% of 58 errors. See `confusion_matrix.png`.

Rationale-contradicts-ranking heuristic (top-1 rationale matches e.g. "argues against", "unlikely"): 0.0% (0 trials).

## Length-normalized ranking (SPEC 15)

| Ranking | Top-1 | Top-3 |
|---|---|---|
| sum logprob | 71.0% | 88.0% |
| logprob / n_answer_tokens | 72.0% | 83.5% |

Top-1 agreement between the two rankings: 89.0% (n = 200).

## Error analysis
annotations not provided.

## Success criteria (SPEC 0.3)

| Criterion | Target | Value | Pass |
|---|---|---|---|
| study_set_size | 200 trials, 10 per country | 200 trials, 40-40 per country over 5 countries | FAIL |
| top1_accuracy | 0.45-0.75 | 0.71 | PASS |
| top3_accuracy | >= 0.75 | 0.88 | PASS |
| human_catchable_share | >= 0.40 of top-1 errors with human_catchable=yes | None | n/a |
