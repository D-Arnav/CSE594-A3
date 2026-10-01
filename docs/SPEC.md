# GeoGuess-HAI — Technical Specification

Human-AI country geolocation from street-level images, with a small local VLM proposing ranked guesses and evidence.

Version 1.0 · Covers Assignment 2 (task, dataset, model, evaluation) and Assignment 3 (study interface)

---

## 0. Overview

### 0.1 What we are building

A participant sees a street-level photo and must decide which country it was taken in, choosing from a fixed list of 20 countries.

The AI is a 2–3B vision-language model (VLM) running locally. It outputs a **ranked top-3 list of countries**, each with a **confidence score** and a **one-to-two sentence rationale citing visible evidence**. In the assisted condition, the human reviews the evidence, checks it against the image, and selects one of the three options or overrides with "Other".

### 0.2 Why this task fits the assignment

The task is built for complementarity. The AI is good at **candidate generation**: it has broad exposure to regional visual patterns (road markings, vegetation, architecture, script) and can narrow 20 countries to a plausible shortlist. Most people can't do this reliably.

Humans are good at **verification**. Small VLMs hallucinate evidence ("cars drive on the left", "Cyrillic sign") and misread text. A human can check a specific visual claim in seconds. Humans also bring world knowledge the small model lacks, like recognizing a familiar storefront chain or a license plate format.

Humans **should not delegate** to the AI. A 2–3B model is expected to reach only ~40–65% top-1 accuracy, and its rationales can sound confident even when they're wrong. The error analysis (Section 8) must show that a meaningful share of its errors are human-detectable.

### 0.3 Success criteria for Assignment 2

| Criterion | Target |
|---|---|
| Study set size | 200 trials, 10 per country, label-balanced |
| AI top-1 accuracy on study set | 45–75% (adjust via Section 9 if outside) |
| AI top-3 accuracy on study set | ≥ 75% (otherwise the human-AI ceiling is too low) |
| Human-detectable share of AI errors | ≥ 40% of top-1 errors fall into catchable categories |
| Reproducibility | GSI can run `predict.py` + `evaluate.py` and match reported numbers |

---

## 1. Task Definition

### 1.1 Trial

| Field | Value |
|---|---|
| Input | One street-level image (no metadata, no EXIF) |
| Label space | 20 countries (Section 4.2) |
| Expected response | One country from the label space |
| Ground truth | Country from the dataset's GPS metadata |
| Target time per trial | 20–40 seconds |

### 1.2 Conditions

These are needed for the Assignment 2 prediction and implemented in Assignment 3. All three use the same images and label space.

| Condition | What the decision-maker sees | Response |
|---|---|---|
| **AI-only** | — | Model's top-1 country (computed offline) |
| **Human-only** | Image + 20-country selector | One of 20 countries |
| **Human-AI** | Image + AI top-3 cards (country, confidence, rationale) + "Other" button revealing the 20-country selector | One of 20 countries |

Do **not** show human-only participants the AI's top-3 without rationale. The candidate list is itself AI assistance and would contaminate the baseline.

### 1.3 Scoring

Country-level exact match. The primary metric is accuracy. A secondary, optional metric is "same macro-region" partial credit (Section 7.4). It's only for analysis, never shown to participants.

---

## 2. System Architecture

```
                ┌───────────────────────┐
  OSV-5M  ───▶  │  build_dataset.py     │ ──▶ data/study/  (200 imgs + study_meta.csv)
  (or own       │  sample, filter, OCR  │ ──▶ data/dev/    (200 imgs, prompt tuning)
   images)      └───────────────────────┘ ──▶ data/train/  (optional, fine-tuning)
                           │
                           ▼
                ┌───────────────────────┐
                │  predict.py           │  Stage 1: likelihood scoring over 20 countries
                │  (src/model.py)       │  Stage 2: rationale generation for top-3
                └───────────────────────┘
                           │
                           ▼
              outputs/study_dataset.csv  ◀── submitted dataset file
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
  ┌──────────────────┐            ┌──────────────────┐
  │ evaluate.py      │            │ annotate.py      │  manual error labels
  │ metrics, ECE,    │ ◀───────── │ (Streamlit)      │  → outputs/annotations.csv
  │ error table      │            └──────────────────┘
  └──────────────────┘
           │
           ▼
  outputs/report/ (metrics.json, figures, error_table.md)

  Assignment 3:
  ┌──────────────────┐   optional live   ┌──────────────────┐
  │ study_app/       │ ────────────────▶ │ server.py        │
  │ (Streamlit)      │                   │ (FastAPI + VLM)  │
  │ reads cached CSV │                   └──────────────────┘
  └──────────────────┘
           │
           ▼
   response log (Google Sheet / SQLite)
```

### 2.1 Repository layout

```
geo-hai/
├── README.md                 # setup + exact commands to reproduce
├── requirements.txt
├── config.yaml               # all tunable parameters (Appendix A)
├── data/
│   ├── raw/                  # downloaded metadata/shards (gitignored)
│   ├── study/                # 200 study images + study_meta.csv
│   ├── dev/                  # 200 dev images + dev_meta.csv
│   └── train/                # optional fine-tuning set
├── src/
│   ├── build_dataset.py
│   ├── model.py              # VLM wrapper: load, score, explain
│   ├── prompts.py            # all prompt templates, versioned
│   ├── predict.py            # batch inference → CSV/JSONL
│   ├── evaluate.py           # metrics + figures + error table
│   ├── annotate.py           # Streamlit annotation tool
│   ├── server.py             # FastAPI live inference
│   └── finetune_lora.py      # optional (+5 bonus path)
├── study_app/                # Assignment 3 interface
│   ├── app.py
│   └── assignments.py        # Latin-square trial allocation
├── notebooks/
│   └── colab_run.ipynb       # one-click Colab pipeline
├── outputs/
│   ├── study_dataset.csv
│   ├── raw_outputs.jsonl
│   ├── annotations.csv
│   └── report/
└── docs/
    ├── iteration_log.md
    └── screenshots/
```

---

## 3. Environment and Compute

| Item | Spec |
|---|---|
| Python | 3.11 |
| Core libs | `torch`, `transformers` (recent enough to support the chosen VLM), `accelerate`, `pillow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `tqdm` |
| Optional libs | `easyocr` (text tagging), `fastapi` + `uvicorn` (server), `streamlit` (annotation + study app), `peft` (LoRA), `datasets`, `huggingface_hub` |
| GPU | Colab T4 (16 GB) is sufficient for a 2–3B model in fp16/bf16. Local Apple Silicon with MPS or 8 GB+ consumer GPU also works. |
| CPU fallback | Possible but slow. Use only for the study app, which reads cached outputs. |

**Reproducibility rules:** fix `seed` in `config.yaml`, use greedy decoding (`do_sample=False`), pin package versions in `requirements.txt`, and log the model ID plus its exact Hugging Face revision hash in every output file.

---

## 4. Dataset Pipeline (`build_dataset.py`)

### 4.1 Source

**Default:** OpenStreetView-5M (OSV-5M), the Mapillary-derived street-view dataset on Hugging Face (`osv5m/osv5m`). Use the **test split** for study and dev sets and the **train split** only for optional fine-tuning. Check the dataset card for current metadata fields, image shard layout, and license (CC-BY-SA), and cite it in the report.

**Bonus (+3) path:** assemble images yourself via the Mapillary API (sample random image IDs inside country bounding boxes, download, and keep GPS for labels) or your own photos. If you take this path, document collection, filtering, and labeling in the report. Confirm with the GSI that this counts as "creating your own dataset."

### 4.2 Label space: 20 countries

The list is chosen so that several **confusable clusters** exist (good for error analysis) while staying regionally diverse:

| Cluster | Countries |
|---|---|
| North America | United States, Canada, Mexico |
| South America | Brazil, Argentina, Chile |
| British Isles | United Kingdom, Ireland |
| Western Europe | France, Spain, Italy, Germany |
| Eastern Europe | Russia, Poland |
| East Asia | Japan, South Korea |
| South / Southeast Asia | India, Thailand |
| Southern Hemisphere | South Africa, Australia |

`build_dataset.py` must verify each country has ≥ 300 candidate images in the source split. If not, it substitutes a country from a backup list in `config.yaml` (e.g., New Zealand, Netherlands, Indonesia, Turkey) and logs the substitution.

### 4.3 Sampling procedure

1. Load test-split metadata and filter to the 20 countries.
2. **Quality filter** (automatic): drop images under 256 px on the short side, and drop near-duplicates by perceptual hash (`imagehash.phash`, Hamming distance ≤ 6).
3. **Geographic spread:** within each country, prefer images ≥ 25 km apart (greedy farthest-point sampling on lat/lon) so a country isn't represented by one street.
4. **Text tagging:** run EasyOCR. Mark `has_text = True` if any detection has confidence ≥ 0.5 and ≥ 3 characters. Store the detected script if identifiable.
5. **Stratify:** for each country, sample **5 images with text + 5 without** for the study set, and do the same, disjointly, for the dev set. If a country lacks enough text images, fill from no-text and log it.
6. **Manual screen** (required): view every study image once. Remove any with clearly identifiable faces, license plates readable as personal data, or a visible country name/flag that makes the trial trivial. Replace from the reserve pool. Mapillary blurs faces and plates, but verify.
7. Copy images to `data/study/` as `S001.jpg … S200.jpg` (ID order shuffled with the seed so IDs leak nothing about country).

### 4.4 Splits and leakage

| Split | Size | Source | Use |
|---|---|---|---|
| `dev` | 200 | OSV-5M test | Prompt engineering, temperature fitting, sanity checks |
| `study` | 200 | OSV-5M test (disjoint from dev) | Evaluation + Assignment 3 trials. Never used for tuning. |
| `train` | 2k–20k | OSV-5M train | Optional LoRA fine-tuning only |

All prompt iteration happens on `dev`. If you change anything after looking at study-set results, record it in `docs/iteration_log.md` and re-run the full evaluation. The assignment allows iteration but requires you to report it.

### 4.5 Output: `study_meta.csv`

| Column | Type | Description |
|---|---|---|
| `trial_id` | str | `S001`–`S200` |
| `image_path` | str | relative path |
| `source_id` | str | original OSV-5M / Mapillary ID (for attribution) |
| `country` | str | ground truth |
| `cluster` | str | from 4.2 |
| `lat`, `lon` | float | kept for analysis only, never shown |
| `has_text` | bool | OCR tag |
| `ocr_text` | str | raw OCR string (analysis only) |

---

## 5. Model and Inference (`model.py`, `predict.py`)

### 5.1 Model choice

| Role | Model (verify exact HF ID and license before use) | Why |
|---|---|---|
| **Primary** | Qwen3-VL-2B-Instruct | Apache 2.0, strong text reading, good `transformers` support |
| Alternative | LiquidAI/LFM2.5-VL-3B | Recent, edge-optimized 3B |
| Weaker fallback | SmolVLM2-2.2B-Instruct | Use if the primary is too accurate (Section 9) |

`config.yaml` holds `model_id` and `revision`. All code must work with any `AutoModelForImageTextToText` + `AutoProcessor` model that has a chat template.

Image resolution: cap visual tokens (e.g., `max_pixels ≈ 512×512` for Qwen-family processors) so 20-candidate scoring fits on a T4. Record the setting in output metadata.

### 5.2 Two-stage pipeline

Small VLMs are unreliable at producing valid JSON with sensible probabilities. So the pipeline separates **ranking** (computed from model likelihoods, always valid and well-defined) from **explanation** (free text, short).

#### Stage 1: Likelihood scoring over all 20 countries

For each image and each candidate country *c*, build a full conversation:

- **User:** image + `PROMPT_RANK` (Section 5.3)
- **Assistant:** `"{c}."`

Then compute the summed log-probability of the assistant tokens `"{c}."` plus the end-of-turn token. The end-of-turn token prevents prefix bias (e.g., "Ireland" scoring partial credit for "Ireland, Northern…").

```python
def score(img, cands):
    # returns dict country -> sum log p(answer tokens | image, prompt)
    lp = {}
    for c in cands:
        msgs = build_msgs(img, PROMPT_RANK, answer=f"{c}.")
        ids = processor.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                            return_tensors="pt").to(dev)
        n_prompt = prompt_len(img)            # length with add_generation_prompt=True
        with torch.no_grad():
            logits = model(**ids).logits[0, :-1]
        tgt = ids["input_ids"][0, 1:]
        logp = logits.log_softmax(-1).gather(-1, tgt[:, None])[:, 0]
        lp[c] = logp[n_prompt - 1:].sum().item()
    return lp
```

Convert to a probability distribution with a temperature *T*:

> p(c) = softmax( logp(c) / T ) over the 20 candidates

- *T* = 1.0 by default. Optionally fit *T* on the **dev** set by minimizing negative log-likelihood (Guo et al., 2017), then freeze it.
- Top-3 = the three highest p(c). Confidence shown to humans = p(c), rounded to 2 decimals.
- Randomize candidate order in the prompt per image (seeded) to reduce position bias. Log the order.

**Efficiency (optional):** the image and prompt prefix is identical across the 20 candidates. Cache `past_key_values` from one prefix pass and score candidates in batches of 5–10. The naive version (200 × 20 forward passes) takes roughly 15–30 minutes on a T4, which is acceptable.

#### Stage 2: Rationale generation for top-3

For each of the top-3 countries, generate a short rationale with `PROMPT_EXPLAIN`, using greedy decoding and `max_new_tokens = 60`. Post-process by stripping whitespace, truncating to two sentences, and replacing an empty output with "No distinctive clues identified."

Runtime: 600 generations, about 10–20 minutes on a T4.

### 5.3 Prompts (`prompts.py`, versioned)

```text
PROMPT_RANK (v1)
You are shown a street-level photo. It was taken in exactly one of these countries:
{shuffled_country_list}
Which country was it taken in? Answer with the country name only.

PROMPT_EXPLAIN (v1)
Consider whether this photo could have been taken in {country}.
In at most two short sentences, name specific things VISIBLE in the image
that support or argue against {country} (e.g., road markings, which side
cars drive on, signs and their script, vegetation, architecture, poles).
Only mention things you can actually see. If there are no useful clues,
say "No distinctive clues."
```

Every prompt version change must be recorded with date, reason, and dev-set effect in `docs/iteration_log.md`.

### 5.4 Generative baseline (for the iteration log, not the study)

Run the model once on the dev set with a free-form JSON prompt asking for top-3 guesses plus confidence plus reasoning. Report the JSON parse-failure rate and accuracy. This justifies the two-stage design in the report. The expected result is that small models often produce invalid JSON, guesses outside the label set, or confidences that don't sum sensibly.

### 5.5 `predict.py` outputs

**`outputs/raw_outputs.jsonl`** has one line per trial:

```json
{"trial_id": "S017", "model_id": "...", "revision": "...", "prompt_versions": {"rank": "v1", "explain": "v1"},
 "temperature": 1.0, "candidate_order": ["..."], "logprobs": {"Chile": -2.1, "...": 0},
 "probs": {"Chile": 0.41, "...": 0},
 "top3": [{"country": "Chile", "p": 0.41, "rationale": "..."},
          {"country": "Argentina", "p": 0.27, "rationale": "..."},
          {"country": "Spain", "p": 0.09, "rationale": "..."}],
 "runtime_s": 6.3}
```

**`outputs/study_dataset.csv`** is the submitted dataset file (Section 6).

CLI:

```
python src/predict.py --split study --config config.yaml
python src/predict.py --split dev --config config.yaml --fit-temperature
```

### 5.6 Optional: LoRA fine-tuning (+5 bonus path, `finetune_lora.py`)

- **Data:** 100–500 images per country from the OSV-5M **train** split, kept geographically separate from study/dev images. As a safeguard, exclude train images within 5 km of any study or dev image.
- **Format:** same `PROMPT_RANK` user turn, with the assistant answer being the country name.
- **Method:** LoRA (rank 16, alpha 32, dropout 0.05) on the language model attention projections, with the vision encoder frozen. 1–2 epochs, lr 1e-4, bf16/fp16, batch via gradient accumulation.
- **Report:** dev and study accuracy before vs. after. If fine-tuning pushes study top-1 above ~75%, the delegation argument weakens. You can keep the fine-tuned model as an ablation and use a lighter checkpoint for the study.
- **Submit:** training script, training data manifest (image IDs + labels), and config. Confirm with the GSI that fine-tuning qualifies as "training your own model."

---

## 6. Submitted Study Dataset Schema (`study_dataset.csv`)

| Column | Description |
|---|---|
| `trial_id` | S001–S200 |
| `image_path` / `image_url` | input |
| `source_id` | source attribution |
| `country` | **ground truth** |
| `cluster` | confusable group |
| `has_text` | OCR tag |
| `ai_top1`, `ai_top2`, `ai_top3` | ranked countries |
| `ai_p1`, `ai_p2`, `ai_p3` | confidences |
| `ai_rationale1`, `ai_rationale2`, `ai_rationale3` | rationales |
| `ai_correct_top1` | bool |
| `ai_correct_top3` | bool |
| `gt_rank` | rank of true country among 20 (1 = best) |
| `error_category` | from annotation (blank if correct) |
| `rationale_faithful_top1` | from annotation: `faithful` / `partially` / `hallucinated` / `none` |
| `human_catchable` | from annotation: `yes` / `maybe` / `no` (blank if correct) |
| `model_id`, `revision`, `prompt_version` | provenance |

---

## 7. Evaluation (`evaluate.py`)

Run `python src/evaluate.py --pred outputs/study_dataset.csv --ann outputs/annotations.csv`. It writes `outputs/report/metrics.json`, figures, and `error_table.md`.

### 7.1 Primary metrics

| Metric | Definition | Why |
|---|---|---|
| Top-1 accuracy | share of trials where `ai_top1 == country` | AI-only performance; standard in image geolocation benchmarks |
| Top-3 accuracy | share where `country ∈ {top1, top2, top3}` | Ceiling for the human-AI condition without override |
| Mean reciprocal rank | mean of 1 / `gt_rank` | How close misses are |
| Chance baseline | 1/20 = 5% | Context |

Report 95% bootstrap confidence intervals (1,000 resamples over trials) for each.

### 7.2 Calibration

Humans will use the confidence numbers, so calibration is directly relevant to reliance.

- **Expected Calibration Error (ECE):** 10 equal-width bins on top-1 confidence. ECE = Σ_b (n_b / N) · |acc_b − conf_b| (Guo et al., 2017).
- **Reliability diagram:** bar chart of accuracy vs. confidence per bin with the diagonal.
- **Accuracy by confidence tercile:** a table that shows whether high-confidence predictions are actually more reliable.

### 7.3 Breakdowns

- Accuracy by `has_text` (expected: higher with text).
- Per-country accuracy (bar chart) and a 20×20 **confusion matrix** (heatmap). Highlight within-cluster confusions.
- Within-cluster error rate: share of top-1 errors where prediction and truth share a cluster.

### 7.4 Secondary

- Same-cluster partial credit accuracy (analysis only).
- Rationale faithfulness rate, from annotations (Section 8.2).

---

## 8. Error Analysis

### 8.1 Taxonomy

Each top-1 error gets exactly one **primary** category:

| Code | Category | Description | Likely human-catchable? |
|---|---|---|---|
| HAL | Hallucinated evidence | Rationale cites something not in the image | Yes |
| TXT | Text misread / ignored | Legible text or script points elsewhere, and the model missed or misread it | Yes |
| NBR | Neighbor confusion | Truth and prediction in the same cluster with ambiguous cues | Maybe |
| CUE | Cue-poor image | Little distinctive evidence. A human would also struggle | No |
| KNW | Knowledge gap | Correct cue noticed but mapped to the wrong country | Maybe |
| OVR | Overconfident miss | Top-1 confidence ≥ 0.6 and wrong (secondary tag; can co-occur) | — (affects reliance) |

### 8.2 Annotation protocol (`annotate.py`)

The rater is you, and optionally a second rater on a 25% subset to report agreement (Cohen's κ).

For each trial where the AI is wrong at top-1, the tool shows the image, ground truth, and the AI's top-3 with rationales. The rater records:

1. **Primary error category** (HAL/TXT/NBR/CUE/KNW), using the first rule that applies in this order: HAL → TXT → NBR → KNW → CUE.
2. **Rationale faithfulness** for the top-1 rationale:
   - **faithful:** every visual claim is visible in the image
   - **partially:** at least one claim is visible and at least one is not
   - **hallucinated:** no claim is visible in the image
   - **none:** the rationale says there are no distinctive clues
3. **Human-catchable:** could an attentive non-expert, given the AI output, reasonably reject the top-1? Answer yes, maybe, or no.
4. Free-text note.

Also annotate faithfulness on a random 50 **correct** trials, so you can report whether right answers come with real evidence or lucky rationalizations.

### 8.3 Error distribution table (submission requirement)

Generated by `evaluate.py`:

| Category | # errors | % of errors | Human-catchable (yes / maybe / no) | Example trial IDs |
|---|---|---|---|---|
| HAL | | | | |
| TXT | | | | |
| NBR | | | | |
| KNW | | | | |
| CUE | | | | |
| *OVR (secondary)* | | | | |

The written discussion should argue complementarity from this table. HAL + TXT errors are the case for "don't delegate", while CUE errors bound what anyone can achieve.

---

## 9. Iteration Loop

Evaluate on the dev set first. Touch the study set only once the pipeline is stable.

| Observation | Lever (in order of preference) |
|---|---|
| Top-1 > 75% (too easy → delegate) | Raise no-text share to 7/10 per country → switch to weaker model → add more confusable countries |
| Top-1 < 45% (unhelpful) | Raise text share → reduce label space to 12–15 countries → switch to stronger model → fine-tune (5.6) |
| Top-3 < 75% | Same as "too hard", plus consider top-3 with mandatory "Other" in the study |
| Labels skewed | Re-run stratified sampling (should not happen by design) |
| Rationales generic / empty | Revise `PROMPT_EXPLAIN` on dev, cap image tokens higher |

`docs/iteration_log.md` gets one entry per change: date, what changed, why, and before/after dev metrics, plus study metrics if re-run. The report summarizes this log (Step J).

---

## 10. Live Inference Service (`server.py`)

This exists to demonstrate real-time AI (the assignment penalizes chat-interface-only outputs) and to enable an optional live mode in the study app.

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/health` | GET | — | `{"status": "ok", "model_id": ..., "revision": ...}` |
| `/predict` | POST | multipart image or `{"trial_id": "S017"}` | same structure as one `raw_outputs.jsonl` line |

- Loads the model once at startup. Uses the same `model.py` functions and config as `predict.py`, so outputs for a study image must match the cached CSV. Add a test that checks this on 5 images.
- Run in Colab with a tunnel (e.g., ngrok or cloudflared) or locally with `uvicorn src.server:app`.
- Expected latency: ~5–10 seconds per image on a T4. This is why the study app defaults to cached outputs.

Screenshot the `/predict` response for one trial as the required "evidence of model running."

---

## 11. Study Interface (Assignment 3, `study_app/`)

### 11.1 Stack

Streamlit app that reads `study_dataset.csv` and images. It needs no GPU, so it can be hosted on Streamlit Community Cloud or run locally with a tunnel. Responses are logged to a Google Sheet (via service account) or SQLite. Streamlit Cloud's filesystem is not persistent, so a local file alone won't work there.

### 11.2 Design: within-subjects, counterbalanced

1. **Consent + instructions** (~1 min), including one worked practice trial per condition that is not scored.
2. **Block A** (8 trials), then **Block B** (8 trials). Each block is one condition: human-only or human-AI. Block order alternates by participant ID.
3. **Trial allocation:** images are partitioned into 25 sets of 8. A Latin square in `assignments.py` ensures each image is seen in both conditions across participants and no participant sees an image twice.
4. **Post-survey** (~1 min): trust in AI (1–7), perceived usefulness (1–7), self-rated geography knowledge (1–7), free text.

Total time is about 9–10 minutes. If pilots run long, reduce blocks to 6 trials each.

### 11.3 Trial screen

**Human-only:** image (large, zoomable), 20-country button grid grouped by cluster, and a Submit button.

**Human-AI:**

- The same image is shown.
- Three AI cards, ordered by rank, each showing country, confidence bar plus percent, and rationale.
- One click selects a card.
- An **"None of these — choose another"** link expands the 20-country grid.
- A small banner reads: *"The AI can be wrong and sometimes describes things that aren't in the image. Check its evidence."* Keep the wording fixed across participants.

No feedback on correctness during the study, so learning effects stay small.

### 11.4 Logging schema

| Field | Description |
|---|---|
| `participant_id`, `session_id` | anonymous IDs |
| `block`, `condition`, `trial_order` | design variables |
| `trial_id` | image |
| `response` | chosen country |
| `correct` | bool |
| `chose_rank` | 1/2/3 if an AI card was picked, 0 if "Other" (human-AI only) |
| `ai_top1_correct` | bool, copied from dataset |
| `rt_ms` | time from image load to submit |
| `expanded_other` | bool |
| `timestamp` | ISO 8601 |

### 11.5 Assignment 3 metrics (planned)

- Accuracy by condition, compared against AI-only on the same images.
- **Agreement:** share of human-AI responses equal to AI top-1.
- **Over-reliance:** accepted AI top-1 when it was wrong.
- **Under-reliance:** rejected AI top-1 when it was right.
- **Rescue rate:** AI top-1 wrong but the human chose the correct option (rank 2, rank 3, or "Other").
- Accuracy split by AI error category (from Section 8). This tests whether HAL/TXT errors really get caught more than CUE errors.

---

## 12. Prediction (Assignment 2 report template)

> **Predicted ranking:** Human-AI > AI-only > Human-only.
>
> **Reasoning:** with 20 options, human-only accuracy will be modest (est. 30–45%) because most students lack systematic geolocation knowledge. The AI (est. top-1 X%, top-3 Y% from Section 7) narrows the choices well. In the human-AI condition, humans should recover a share of AI errors that the error analysis marks as catchable (HAL + TXT = Z% of errors).
>
> **Main risk:** over-reliance. Rationales sound fluent and confidences may be miscalibrated (ECE = …), so participants may accept wrong top-1 answers. This would push human-AI toward AI-only. A secondary risk is that the 3-option framing anchors people even when "Other" is available.
>
> **Alternative outcome we consider plausible:** Human-AI ≈ AI-only > Human-only.

Fill in X, Y, Z, and the ECE from actual results.

---

## 13. Assignment 2 Deliverables Map

| Required item | Where it comes from |
|---|---|
| Task description with illustration | Section 1 + screenshot of a study-app mockup or a sample trial with AI cards |
| Dataset link + curation description | Section 4 (OSV-5M link, sampling steps, filters, stratification) |
| AI assistance + model description | Sections 5.1–5.3 + architecture diagram (Section 2) |
| Implementation screenshots | Colab model loading, `predict.py` progress, `/predict` response, one rendered trial output |
| Justification (benefit but don't delegate) | Section 0.2, backed by the error table |
| Metrics + references | Section 7 (Guo et al. 2017 for ECE; OSV-5M and IM2GPS papers for geolocation metrics) |
| Error analysis + distribution table | Section 8 |
| Prediction | Section 12 |
| Iterations | `docs/iteration_log.md` summary |
| Study dataset file | `outputs/study_dataset.csv` (Section 6) |
| Model code + instructions | `src/model.py`, `src/predict.py`, `README.md`, `notebooks/colab_run.ipynb` |
| Evaluation code | `src/evaluate.py` + `annotations.csv` |

The README must include a **"Reproduce in 3 commands"** section:

```
pip install -r requirements.txt
python src/predict.py --split study --config config.yaml
python src/evaluate.py --pred outputs/study_dataset.csv --ann outputs/annotations.csv
```

Also provide a path that skips inference entirely, so the GSI can verify metrics from the submitted CSV alone with only the evaluation command.

---

## 14. Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Environment + model smoke test | Model loads in Colab and answers for 1 image |
| 2 | Dataset built | `study/` and `dev/` folders + meta CSVs, manual screen complete |
| 3 | Stage 1 scoring on dev | Top-1/top-3 on dev reported, temperature fitted |
| 4 | Stage 2 rationales on dev | Rationales spot-checked on 20 images |
| 5 | Iteration (if needed) | Dev metrics within Section 0.3 targets, log written |
| 6 | Study-set inference | `study_dataset.csv` complete |
| 7 | Annotation | All top-1 errors labeled, plus 50 correct trials |
| 8 | Evaluation + figures | `metrics.json`, reliability diagram, confusion matrix, error table |
| 9 | Server + screenshots | `/predict` working, screenshots captured |
| 10 | Report + submission | PDF + CSV + code zipped |
| 11 | (A3) Study app + pilot | 2 pilot participants, timing confirmed under 10 min |

---

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Model too weak (top-1 near chance) | Text-heavy sampling, smaller label space, stronger model (Section 9) |
| Model too strong (people would delegate) | Cue-poor sampling, weaker model |
| Position bias in scoring | Shuffle candidate order per image; optionally average scores over 2–3 orderings |
| Multi-token name bias in likelihoods | End-of-turn token included; compare with length-normalized scoring on dev and report |
| Rationales contradict ranking ("argues against Chile" as top-1) | Acceptable and informative for humans; note the rate in the report |
| Images reveal country trivially (flags, country names) | Manual screen in 4.3 step 6 |
| Privacy | Mapillary blurring + manual screen; no people-focused images |
| Colab session timeouts | `predict.py` writes JSONL incrementally and resumes from the last completed `trial_id` |
| GSI cannot run GPU code | Cached outputs + evaluation-only path |

---

## Appendix A: `config.yaml`

See `/config.yaml` at repo root (extended with a few extra data knobs and a pinned revision).

## Appendix B: References to cite

- Astruc et al. (2024). *OpenStreetView-5M: The Many Roads to Global Visual Geolocation.* CVPR.
- Hays & Efros (2008). *IM2GPS: Estimating Geographic Information from a Single Image.* CVPR.
- Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
- Model card for the chosen VLM (cite exact HF ID and revision).
- Optional HAI framing: Bansal et al. (2021), *Does the Whole Exceed Its Parts? The Effect of AI Explanations on Complementary Team Performance*, CHI. Directly relevant to explanations and over-reliance.
