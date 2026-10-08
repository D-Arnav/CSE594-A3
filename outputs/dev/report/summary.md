# Evaluation report

Predictions: `outputs/dev/dev_dataset.csv` · model `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4` · n = 200 · 1000 bootstrap resamples (seed 42)

## Primary metrics (95% bootstrap CI)

| Metric | Value |
|---|---|
| Top-1 accuracy | 72.5% [66.5%, 78.0%] |
| Top-3 accuracy | 87.0% [82.5%, 91.5%] |
| Mean reciprocal rank | 0.824 [0.785, 0.862] |
| Same-cluster partial credit | 78.0% [72.5%, 83.5%] |
| Chance baseline | 20.0% |

## Calibration

ECE (10 bins) = 0.0865. See `reliability.png`.

| Tercile | n | ai_p1 range | mean ai_p1 | Top-1 acc |
|---|---|---|---|---|
| low | 65 | 0.290–0.520 | 0.421 | 33.8% |
| mid | 68 | 0.530–0.840 | 0.678 | 83.8% |
| high | 67 | 0.850–1.000 | 0.937 | 98.5% |

## Accuracy by has_text

| has_text | n | Top-1 | Top-3 |
|---|---|---|---|
| False | 180 | 73.3% | 86.7% |
| True | 20 | 65.0% | 90.0% |

## Per-country accuracy

| Country | Cluster | n | Top-1 | Top-3 |
|---|---|---|---|---|
| United States | North America | 40 | 97.5% | 100.0% |
| Germany | Western Europe | 40 | 22.5% | 47.5% |
| Japan | East Asia | 40 | 95.0% | 100.0% |
| South Korea | East Asia | 40 | 70.0% | 100.0% |
| India | South / Southeast Asia | 40 | 77.5% | 87.5% |

Within-cluster error rate (share of top-1 errors where prediction and truth share a cluster): 20.0% of 55 errors. See `confusion_matrix.png`.

Rationale-contradicts-ranking heuristic (top-1 rationale matches e.g. "argues against", "unlikely"): 0.0% (0 trials).

## Length-normalized ranking (SPEC 15)

| Ranking | Top-1 | Top-3 |
|---|---|---|
| sum logprob | 72.5% | 87.0% |
| logprob / n_answer_tokens | 71.0% | 84.5% |

Top-1 agreement between the two rankings: 88.0% (n = 200).

## Error analysis
annotations not provided.

## Success criteria (SPEC 0.3)

| Criterion | Target | Value | Pass |
|---|---|---|---|
| study_set_size | 200 trials, 10 per country | 200 trials, 40-40 per country over 5 countries | FAIL |
| top1_accuracy | 0.45-0.75 | 0.725 | PASS |
| top3_accuracy | >= 0.75 | 0.87 | PASS |
| human_catchable_share | >= 0.40 of top-1 errors with human_catchable=yes | None | n/a |
