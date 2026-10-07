# Evaluation report

Predictions: `outputs/study_dataset.csv` · model `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203` · n = 200 · 1000 bootstrap resamples (seed 42)

## Primary metrics (95% bootstrap CI)

| Metric | Value |
|---|---|
| Top-1 accuracy | 59.0% [52.0%, 65.5%] |
| Top-3 accuracy | 81.0% [75.0%, 86.0%] |
| Mean reciprocal rank | 0.720 [0.668, 0.767] |
| Same-cluster partial credit | 75.0% [69.0%, 80.0%] |
| Chance baseline | 5.0% |

## Calibration

ECE (10 bins) = 0.0516. See `reliability.png`.

| Tercile | n | ai_p1 range | mean ai_p1 | Top-1 acc |
|---|---|---|---|---|
| low | 66 | 0.100–0.450 | 0.295 | 30.3% |
| mid | 63 | 0.460–0.820 | 0.631 | 57.1% |
| high | 71 | 0.830–0.990 | 0.930 | 87.3% |

## Accuracy by has_text

| has_text | n | Top-1 | Top-3 |
|---|---|---|---|
| False | 191 | 58.1% | 80.6% |
| True | 9 | 77.8% | 88.9% |

## Per-country accuracy

| Country | Cluster | n | Top-1 | Top-3 |
|---|---|---|---|---|
| United States | North America | 10 | 70.0% | 90.0% |
| Canada | North America | 10 | 60.0% | 90.0% |
| Mexico | North America | 10 | 70.0% | 80.0% |
| Brazil | South America | 10 | 50.0% | 90.0% |
| Argentina | South America | 10 | 10.0% | 70.0% |
| Chile | South America | 10 | 30.0% | 70.0% |
| United Kingdom | British Isles | 10 | 10.0% | 70.0% |
| Ireland | British Isles | 10 | 70.0% | 70.0% |
| France | Western Europe | 10 | 40.0% | 70.0% |
| Spain | Western Europe | 10 | 70.0% | 100.0% |
| Italy | Western Europe | 10 | 50.0% | 90.0% |
| Germany | Western Europe | 10 | 50.0% | 60.0% |
| Russia | Eastern Europe | 10 | 70.0% | 80.0% |
| Poland | Eastern Europe | 10 | 20.0% | 60.0% |
| Japan | East Asia | 10 | 80.0% | 90.0% |
| South Korea | East Asia | 10 | 90.0% | 90.0% |
| India | South / Southeast Asia | 10 | 90.0% | 100.0% |
| Thailand | South / Southeast Asia | 10 | 80.0% | 80.0% |
| South Africa | Southern Hemisphere | 10 | 70.0% | 70.0% |
| Australia | Southern Hemisphere | 10 | 100.0% | 100.0% |

Within-cluster error rate (share of top-1 errors where prediction and truth share a cluster): 39.0% of 82 errors. See `confusion_matrix.png`.

Rationale-contradicts-ranking heuristic (top-1 rationale matches e.g. "argues against", "unlikely"): 0.0% (0 trials).

## Length-normalized ranking (SPEC 15)

| Ranking | Top-1 | Top-3 |
|---|---|---|
| sum logprob | 59.0% | 81.0% |
| logprob / n_answer_tokens | 59.5% | 82.5% |

Top-1 agreement between the two rankings: 98.5% (n = 200).

## Error analysis
annotations not provided.

## Success criteria (SPEC 0.3)

| Criterion | Target | Value | Pass |
|---|---|---|---|
| study_set_size | 200 trials, 10 per country | 200 trials, 10-10 per country over 20 countries | PASS |
| top1_accuracy | 0.45-0.75 | 0.59 | PASS |
| top3_accuracy | >= 0.75 | 0.81 | PASS |
| human_catchable_share | >= 0.40 of top-1 errors with human_catchable=yes | None | n/a |
