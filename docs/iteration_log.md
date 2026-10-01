# Iteration log

This log has one entry per change, following SPEC 9. All tuning happens on **dev** (`D001`–`D200`). The study
split is evaluated only once the pipeline is stable. Any change made after study results were seen gets an entry
here, and the full study evaluation is then re-run.

Getting the dev metrics:

```bash
python src/predict.py --split dev --config config.yaml --fit-temperature
python src/evaluate.py --pred outputs/dev/dev_dataset.csv --raw outputs/dev/raw_outputs.jsonl --out outputs/dev/report
```

Targets (SPEC 0.3) are top-1 between 45% and 75%, top-3 of at least 75%, and at least 40% of top-1 errors
human-catchable. Use the levers in SPEC 9 in the order listed.

Template for new entries:

```
## <n>. <short title> — YYYY-MM-DD
- Changed:  <config key / prompt version / code>
- Why:      <observation that triggered it>
- Dev before -> after: top-1 _, top-3 _, MRR _, ECE _ (T _)
- Study (only if re-run): top-1 _, top-3 _, MRR _, ECE _
- Decision: keep / revert
```

---

## 1. Initial configuration (v1) — 2026-10-01

**Changed:** nothing. This is the starting point.

- Model: `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`, `float16`, greedy decoding,
  `max_pixels: 262144` (about 512×512), `seed: 42`.
- Label space: the 20 countries of SPEC 4.2 (no substitutions unless `data/countries.txt` exists).
- Prompts: `rank=v1`, `explain=v1` (SPEC 5.3 text, in `src/prompts.py`).
- Stage 1 scoring: the summed log p of the answer tokens plus the end-of-turn token. The candidate order is
  shuffled with a seed per image (`shuffle_candidates: true`, `n_orderings: 1`). The image+prompt prefix is cached
  once and candidates are scored in batches of 5 (`score_batch_size`). `--naive` is the reference path.
- **Deviation from SPEC 5.2:** the scored answer is `"{c}"`, not `"{c}."` (`RANK_ANSWER` in `prompts.py`).
  Qwen3-VL-2B answers this prompt with the bare name. Its log p for the trailing period is about −20 and varies by
  1–2 nats across candidates, which is noise unrelated to the image.
- Stage 2: `explain_max_new_tokens: 60`. The output is cut to two sentences, and an empty one becomes
  "No distinctive clues identified."
- Temperature: T = 1.0 in config, then fitted on dev with `--fit-temperature` → `outputs/temperature.json`.
- **Deviation from SPEC 4.3 step 5:** `data.text_ratio: null`, so there is no 5 text + 5 no-text stratification.
  EasyOCR (`ocr_conf 0.5`, `ocr_min_chars 3`) flags very few OSV-5M images as having text; the counts are in the
  `TEXT <country>:` lines of `data/build_log.txt`: 35 OCR-text out of 1,198 candidates, mostly dashcam
  overlays, not street signs. `has_text` is kept as the OCR tag for analysis only.

**Why:** this is the SPEC default design.

**Dev metrics (n = 200, MPS fp16, `max_pixels` 262144):**

| Metric | T = 1.0 | T fitted = 2.869 |
|---|---|---|
| Top-1 accuracy | 60.5% [54.0, 67.5] | same (T does not change the ranking) |
| Top-3 accuracy | 79.5% [74.0, 85.0] | same |
| MRR | 0.724 | same |
| NLL | 2.324 | 1.435 |
| ECE (10 bins) | 0.224 (mean top-1 p 0.83 vs 60.5% acc) | 0.082 |
| Top-1 acc, has_text = True / False | 80% (n = 5) / 60% (n = 195) | |
| Mean s/trial (score / explain), device | 1.07 / 4.16, MPS | |

Confidence terciles after T: low 35.4%, mid 57.6%, high 87.0% top-1 accuracy, so higher confidence is more
reliable. Within-cluster share of top-1 errors: 39.2% of 79. Weakest: Argentina 20%, Ireland 30%,
South Africa 30%; strongest: India and Australia 100%, United Kingdom 90%.

**Image cap check.** A bug left the first partial run without the `max_pixels` cap (fixed in `3a86c7a`). On the
same 78 dev trials, full resolution gave top-1 66.7% / top-3 82.1%, and the capped run 60.3% / 79.5%. Both
are within target. The capped setting is kept because SPEC 5.1 asks for it, and it halves the image tokens.

**Rationale spot-check (milestone 4, all 200 dev trials):**

| Rationale | "No distinctive clues" | Opens with "Yes/No, this photo could…" |
|---|---|---|
| Top-1 | 42.0% | 55.5% |
| Top-2 | 86.5% | 11.5% |
| Top-3 | 98.5% | 1.0% |

No top-1 rationale contradicts the ranking ("argues against", "unlikely": 0%).

**Decision:** ranking and calibration are within the SPEC 0.3 targets, so Stage 1 is frozen. The rationales are
too often empty to support verification in the human-AI condition. Per SPEC 9 the next lever is revising
`PROMPT_EXPLAIN` on dev (entry 2).

---

## JSON generative baseline (SPEC 5.4)

```bash
python src/predict.py --split dev --config config.yaml --baseline-json
```

The same image and shuffled candidate list are used, with the model asked for JSON containing top-3 guesses,
confidences and reasoning (`PROMPT_JSON_BASELINE` v1, `baseline_max_new_tokens: 512`). The numbers come from
`outputs/dev/baseline_json_summary.json`.

| Metric | Value |
|---|---|
| n | _[TODO]_ |
| Parse-failure rate | _[TODO]_ |
| Truncated at token cap | _[TODO]_ |
| Not exactly 3 guesses (of parsed) | _[TODO]_ |
| Guesses outside the label set | _[TODO]_ |
| Missing confidence (of parsed) | _[TODO]_ |
| Mean confidence sum / share within 0.05 of 1 / share > 1.05 | _[TODO]_ / _[TODO]_ / _[TODO]_ |
| Top-1 / top-3 accuracy (parsed only) | _[TODO]_ / _[TODO]_ |
| Top-1 / top-3 accuracy (all, failures count as wrong) | _[TODO]_ / _[TODO]_ |
| Two-stage pipeline, same dev set: top-1 / top-3 | _[TODO]_ / _[TODO]_ |

**Takeaway:** _[TODO: does this justify likelihood ranking over free-form JSON?]_

---

## Length-normalized vs summed scoring (SPEC 15)

The concern is that summed log p penalizes country names that split into more tokens, such as
"United Kingdom" and "South Africa". `evaluate.py --raw` re-ranks from the stored `logprobs` and
`n_answer_tokens`, using `logprob / n_answer_tokens`, so no new inference is needed. The numbers come from
`length_normalized` in `outputs/dev/report/metrics.json`.

| Scoring | Top-1 | Top-3 |
|---|---|---|
| Summed log p (used) | 60.5% | 79.5% |
| Length-normalized | 60.5% | 79.0% |
| Top-1 agreement between the two | 98.0% | |

Per-country effect (multi-token names): the two rankings disagree on top-1 in only 4 of 200 trials (D030, D063,
D081, D165). Normalizing gains one United States hit (D063) and loses one Brazil hit (D081). Every candidate is 1–3
tokens, so length has little room to bias the sum.

**Decision:** keep summed log p (SPEC default); normalizing changes nothing material.
