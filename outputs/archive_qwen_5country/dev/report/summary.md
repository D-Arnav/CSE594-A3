# Evaluation report

Predictions: `outputs/dev/dev_dataset.csv` · model `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` · n = 200 · 1000 bootstrap resamples (seed 42)

## Primary metrics (95% bootstrap CI)

| Metric | Value |
|---|---|
| Top-1 accuracy | 97.5% [95.0%, 99.5%] |
| Top-3 accuracy | 99.5% [98.5%, 100.0%] |
| Mean reciprocal rank | 0.985 [0.971, 0.997] |
| Same-cluster partial credit | 99.0% [97.5%, 100.0%] |
| Chance baseline | 20.0% |

## Calibration

ECE (10 bins) = 0.0190. See `reliability.png`.

| Tercile | n | ai_p1 range | mean ai_p1 | Top-1 acc |
|---|---|---|---|---|
| low | 47 | 0.320–0.990 | 0.867 | 89.4% |
| high | 153 | 1.000–1.000 | 1.000 | 100.0% |

## Accuracy by has_text

| has_text | n | Top-1 | Top-3 |
|---|---|---|---|
| False | 180 | 97.2% | 99.4% |
| True | 20 | 100.0% | 100.0% |

## Per-country accuracy

| Country | Cluster | n | Top-1 | Top-3 |
|---|---|---|---|---|
| United States | North America | 40 | 100.0% | 100.0% |
| Germany | Western Europe | 40 | 100.0% | 100.0% |
| Japan | East Asia | 40 | 100.0% | 100.0% |
| South Korea | East Asia | 40 | 87.5% | 97.5% |
| India | South / Southeast Asia | 40 | 100.0% | 100.0% |

Within-cluster error rate (share of top-1 errors where prediction and truth share a cluster): 60.0% of 5 errors. See `confusion_matrix.png`.

Rationale-contradicts-ranking heuristic (top-1 rationale matches e.g. "argues against", "unlikely"): 0.0% (0 trials).

## Length-normalized ranking (SPEC 15)

| Ranking | Top-1 | Top-3 |
|---|---|---|
| sum logprob | 97.5% | 99.5% |
| logprob / n_answer_tokens | 97.5% | 100.0% |

Top-1 agreement between the two rankings: 100.0% (n = 200).

## Error analysis
annotations not provided.

## Success criteria (SPEC 0.3)

| Criterion | Target | Value | Pass |
|---|---|---|---|
| study_set_size | 200 trials, 10 per country | 200 trials, 40-40 per country over 5 countries | FAIL |
| top1_accuracy | 0.45-0.75 | 0.975 | FAIL |
| top3_accuracy | >= 0.75 | 0.995 | PASS |
| human_catchable_share | >= 0.40 of top-1 errors with human_catchable=yes | None | n/a |
