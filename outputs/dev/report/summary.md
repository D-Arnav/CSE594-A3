# Evaluation report

Predictions: `outputs/dev/dev_dataset.csv` · model `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` · n = 200 · 1000 bootstrap resamples (seed 42)

## Primary metrics (95% bootstrap CI)

| Metric | Value |
|---|---|
| Top-1 accuracy | 60.5% [54.0%, 67.5%] |
| Top-3 accuracy | 79.5% [74.0%, 85.0%] |
| Mean reciprocal rank | 0.724 [0.678, 0.772] |
| Same-cluster partial credit | 76.0% [70.0%, 82.0%] |
| Chance baseline | 5.0% |

## Calibration

ECE (10 bins) = 0.0805. See `reliability.png`.

| Tercile | n | ai_p1 range | mean ai_p1 | Top-1 acc |
|---|---|---|---|---|
| low | 65 | 0.110–0.350 | 0.229 | 35.4% |
| mid | 66 | 0.370–0.820 | 0.580 | 57.6% |
| high | 69 | 0.830–0.990 | 0.933 | 87.0% |

## Accuracy by has_text

| has_text | n | Top-1 | Top-3 |
|---|---|---|---|
| False | 195 | 60.0% | 79.5% |
| True | 5 | 80.0% | 80.0% |

## Per-country accuracy

| Country | Cluster | n | Top-1 | Top-3 |
|---|---|---|---|---|
| United States | North America | 10 | 40.0% | 80.0% |
| Canada | North America | 10 | 50.0% | 80.0% |
| Mexico | North America | 10 | 60.0% | 80.0% |
| Brazil | South America | 10 | 70.0% | 100.0% |
| Argentina | South America | 10 | 20.0% | 50.0% |
| Chile | South America | 10 | 60.0% | 80.0% |
| United Kingdom | British Isles | 10 | 90.0% | 100.0% |
| Ireland | British Isles | 10 | 30.0% | 70.0% |
| France | Western Europe | 10 | 40.0% | 80.0% |
| Spain | Western Europe | 10 | 70.0% | 100.0% |
| Italy | Western Europe | 10 | 60.0% | 70.0% |
| Germany | Western Europe | 10 | 50.0% | 70.0% |
| Russia | Eastern Europe | 10 | 70.0% | 70.0% |
| Poland | Eastern Europe | 10 | 40.0% | 40.0% |
| Japan | East Asia | 10 | 80.0% | 90.0% |
| South Korea | East Asia | 10 | 80.0% | 90.0% |
| India | South / Southeast Asia | 10 | 100.0% | 100.0% |
| Thailand | South / Southeast Asia | 10 | 70.0% | 80.0% |
| South Africa | Southern Hemisphere | 10 | 30.0% | 60.0% |
| Australia | Southern Hemisphere | 10 | 100.0% | 100.0% |

Within-cluster error rate (share of top-1 errors where prediction and truth share a cluster): 39.2% of 79 errors. See `confusion_matrix.png`.

Rationale-contradicts-ranking heuristic (top-1 rationale matches e.g. "argues against", "unlikely"): 0.0% (0 trials).

## Length-normalized ranking (SPEC 15)

| Ranking | Top-1 | Top-3 |
|---|---|---|
| sum logprob | 60.5% | 79.5% |
| logprob / n_answer_tokens | 60.5% | 79.0% |

Top-1 agreement between the two rankings: 98.0% (n = 200).

## Error analysis
annotations not provided.

## Success criteria (SPEC 0.3)

| Criterion | Target | Value | Pass |
|---|---|---|---|
| study_set_size | 200 trials, 10 per country | 200 trials, 10-10 per country over 20 countries | PASS |
| top1_accuracy | 0.45-0.75 | 0.605 | PASS |
| top3_accuracy | >= 0.75 | 0.795 | PASS |
| human_catchable_share | >= 0.40 of top-1 errors with human_catchable=yes | None | n/a |
