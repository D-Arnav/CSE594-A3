# GeoGuess-HAI

Human-AI country geolocation from street-level photos (UMich CSE594, Assignments 2 and 3).
The full design is in [`docs/SPEC.md`](docs/SPEC.md).

**Task.** Someone sees one street-level photo (OSV-5M / Mapillary) and picks the country it was taken in
from a fixed list of 20 countries. These are grouped into 8 confusable clusters, e.g. United Kingdom/Ireland and Japan/South Korea.
Scoring is exact country match.

**AI assistance.** A small local VLM, Qwen3-VL-2B-Instruct, proposes a ranked **top-3** list. Each guess has a
**confidence** and a **1–2 sentence rationale** that cites visible evidence. The pipeline has two stages:

1. *Ranking:* the summed log-likelihood of each country name as the assistant answer to `PROMPT_RANK`, plus the
   end-of-turn token. The candidate order is shuffled with a seed for each image, the scores go through a softmax
   with temperature *T*, and *T* is fitted on the dev split.
2. *Explanation:* one greedy generation with `PROMPT_EXPLAIN` for each top-3 country.

The ranking is always valid and well-defined. Section 5.2 of the SPEC explains why, and the JSON baseline in
`docs/iteration_log.md` is the comparison.

**Conditions** (SPEC 1.2):

| Condition | Decision-maker sees | Response |
|---|---|---|
| AI-only | — | the model's top-1, computed offline |
| Human-only | image + 20-country selector | one of 20 |
| Human-AI | image + 3 AI cards (country, confidence, rationale) + "None of these" which opens the 20-country selector | one of 20 |

## Repository layout

```
config.yaml                # every tunable parameter (seed, countries, model id + pinned revision, inference, eval)
requirements.txt
src/
  common.py                # country/cluster maps, config + seed helpers, paths
  build_dataset.py         # OSV-5M test split -> data/{study,dev}/ (SPEC 4)
  prompts.py               # versioned prompt templates (SPEC 5.3, 5.4)
  model.py                 # VLM wrapper: load, likelihood scoring, explanation, temperature (SPEC 5.1-5.2)
  predict.py               # batch inference -> raw_outputs.jsonl + {split}_dataset.csv; JSON baseline
  annotate.py              # Streamlit error-annotation tool (SPEC 8.2)
  evaluate.py              # metrics, calibration, breakdowns, figures, error table (SPEC 7, 8.3)
  server.py                # FastAPI live inference: /health, /predict (SPEC 10)
  finetune_lora.py         # optional LoRA bonus path (SPEC 5.6)
study_app/
  app.py                   # Assignment 3 Streamlit study interface (SPEC 11)
  assignments.py           # counterbalanced trial allocation
notebooks/colab_run.ipynb  # the whole pipeline on a Colab T4
tests/                     # pytest suite
docs/
  SPEC.md, iteration_log.md, screenshots/
data/                      # built by build_dataset.py (raw/ is a download cache)
outputs/                   # predictions, annotations, report/
```

## Setup

Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The model is `Qwen/Qwen3-VL-2B-Instruct` at revision `89644892e4d85e24eaac8bacfd4f463576704203` (pinned in
`config.yaml`). It downloads from Hugging Face on first use (about 4 GB), and no token is needed.
Run every command below from the repo root.

## Reproduce in 3 commands

```bash
pip install -r requirements.txt
python src/predict.py --split study --config config.yaml
python src/evaluate.py --pred outputs/study_dataset.csv --ann outputs/annotations.csv
```

Requirements:

- `data/study/` (images + `study_meta.csv`) and `outputs/temperature.json` from the submission must be present.
  Without `temperature.json`, `predict.py` uses `inference.temperature` from `config.yaml` (1.0), so the
  confidences will differ.
- `predict.py` resumes from `outputs/raw_outputs.jsonl`. If that file is already complete, the command only
  re-applies the temperature and rewrites the CSV. To re-run inference from scratch, move it out of the way first:
  `mv outputs/raw_outputs.jsonl outputs/raw_outputs.prev.jsonl`.
- `evaluate.py --ann` writes the annotation columns back into `--pred`. Add `--no-merge` to leave the CSV untouched.

### Evaluation only (no GPU)

`outputs/study_dataset.csv` is submitted with the AI outputs and the merged annotation columns. Recomputing every
metric, figure and the error table from it needs no model, no GPU and no images:

```bash
pip install pandas numpy matplotlib pyyaml scikit-learn
python src/evaluate.py --pred outputs/study_dataset.csv
```

`scikit-learn` is only needed for Cohen's κ (`--ann2`). Add `--ann outputs/annotations.csv` to re-merge from
the raw annotation file instead of using the columns already in the CSV.

## Full pipeline

The order matters: all tuning happens on **dev**, and **study** is touched only once the pipeline is stable (SPEC 4.4, 9).

**1. Build the dataset** (SPEC 4). This downloads OSV-5M `test.csv` and only the needed images (random access
into the remote zips), applies the quality filter, phash dedup and 25 km spread, and runs EasyOCR. It then writes
disjoint study and dev sets (10 per country each), with trial IDs shuffled by the seed.

```bash
python src/build_dataset.py --config config.yaml --contact-sheet
```

Outputs: `data/{study,dev}/{S,D}NNN.jpg`, `data/{split}/{split}_meta.csv`, and `data/build_log.txt`, which
records filter counts, OCR text counts per country and any substitution. `data/countries.txt` is written only if
a country had too few candidates and was swapped for a backup.

`data.text_ratio: null` (the default) turns off the SPEC's 5 text + 5 no-text split, because EasyOCR at OSV-5M
resolution finds almost no text (see `docs/iteration_log.md`). `has_text` is the OCR tag. Set `text_ratio: 0.5` to
stratify on it.

**2. Look over the images** (optional). `data/{study,dev}/contact_sheet.html` shows every image with its trial ID,
country and OCR text. There is no manual exclusion step yet (SPEC 4.3 step 6); Mapillary blurs faces and plates.

**3. Dev predictions + temperature fit** (SPEC 5.2, milestone 3):

```bash
python src/predict.py --split dev --config config.yaml --fit-temperature
```

This writes `outputs/dev/raw_outputs.jsonl`, `outputs/dev/dev_dataset.csv` and `outputs/temperature.json`
(`T`, `nll_before`, `nll_after`, `n`, `split`). It prints top-1/top-3/MRR and the mean seconds per trial.
`--limit N` runs only the first N trials. `--naive` uses one full forward pass per candidate instead of the cached
prefix, which is a slow reference. `--overwrite` discards a JSONL produced by a different
model/revision/prompt version.

Dev evaluation, including the comparison of length-normalized and summed scoring (SPEC 15). Use a separate
`--out` so the study report is not overwritten:

```bash
python src/evaluate.py --pred outputs/dev/dev_dataset.csv --raw outputs/dev/raw_outputs.jsonl --out outputs/dev/report
```

**4. JSON baseline** (SPEC 5.4, for the iteration log only):

```bash
python src/predict.py --split dev --config config.yaml --baseline-json
```

This writes `outputs/dev/baseline_json.jsonl` and `outputs/dev/baseline_json_summary.json`. The summary covers
the parse-failure rate, truncation, guesses outside the label set, confidence sums, and top-1/top-3 accuracy.

**5. Study predictions** (milestone 6). Run this only after the dev metrics are within the SPEC 0.3 targets and
every change is logged in `docs/iteration_log.md`:

```bash
python src/predict.py --split study --config config.yaml
```

This writes `outputs/raw_outputs.jsonl` and `outputs/study_dataset.csv` (SPEC 6 columns, plus `temperature`,
`max_pixels`). The annotation columns stay blank until step 7.

**6. Annotate** (SPEC 8.2). The tool covers every top-1 error plus `--n-correct` correct trials (default 50), for
faithfulness:

```bash
streamlit run src/annotate.py -- --pred outputs/study_dataset.csv --out outputs/annotations.csv --rater me
# optional second rater on a 25% subset, for Cohen's kappa:
streamlit run src/annotate.py -- --pred outputs/study_dataset.csv --out outputs/annotations_rater2.csv --rater r2 --subset-frac 0.25
```

**7. Evaluate** (SPEC 7, 8.3):

```bash
python src/evaluate.py --pred outputs/study_dataset.csv --ann outputs/annotations.csv \
    --ann2 outputs/annotations_rater2.csv --raw outputs/raw_outputs.jsonl
```

**8. Live server** (SPEC 10). It loads the model once, then serves requests through the same `model.py` code path
and temperature as `predict.py`:

```bash
uvicorn src.server:app --port 8000        # or: python src/server.py --port 8000 [--host 0.0.0.0] [--config config.yaml]
curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"trial_id": "S017"}'
curl -X POST localhost:8000/predict -F image=@data/study/S017.jpg
```

`uvicorn` reads its config from `$GEOHAI_CONFIG`, and the default is `config.yaml`. A `/predict` response is one
record in the `raw_outputs.jsonl` format. Take the "model running" screenshot (SPEC 13) from it.

**9. Study app** (Assignment 3, SPEC 11). It needs no GPU and reads `outputs/study_dataset.csv` (plus
`outputs/dev/dev_dataset.csv` for the practice trials):

```bash
streamlit run study_app/app.py
```

- URL parameters: `?pid=N` fixes the participant ID. On Streamlit Cloud the DB is wiped on restart, so give each
  participant an explicit `?pid=N` link (0,1,2,...); auto-numbering is only safe locally. `?live=1` fetches the AI
  outputs from the server at `$GEOHAI_SERVER_URL` and falls back to the cache.
- Responses are logged to `study_app/responses.db` (SQLite). Override with `$GEOHAI_DB`.
- Mirroring to Google Sheets is optional and is set up through `st.secrets` (see the docstring of `study_app/app.py`).

**Optional: LoRA** (SPEC 5.6):

```bash
python src/finetune_lora.py manifest --config config.yaml --per-country 200
python src/finetune_lora.py train --config config.yaml --epochs 1 --out outputs/lora
```

## Compute and runtime

- **Colab T4 (16 GB):** `notebooks/colab_run.ipynb` runs everything, with `model.dtype: float16`. `predict.py`
  appends to the JSONL after every trial and resumes from it after a session timeout.
- **Apple Silicon (MPS, 16 GB):** this works as is, because `model.device: auto` picks cuda, then mps, then cpu.
  Run only one model process at a time (for example, not `predict.py` and `server.py` together), since each one
  loads its own copy of the weights. Normal usage needs no lock or wrapper.
- **CPU:** this falls back to float32 and is slow. Use it only for `evaluate.py`, `annotate.py` and the study app,
  which never load the model.
- If scoring raises `non-finite logprobs`, set `model.dtype` to `bfloat16` (not on a T4) or `float32`.
- Measured runtime: _[TODO: s/trial on T4]_ and _[TODO: s/trial on MPS]_, about _[TODO]_ min for 200 trials. The
  SPEC estimates are 15–30 min for scoring and 10–20 min for rationales on a T4. `predict.py` prints the measured
  `runtime_s`/`score_s`/`explain_s` means.

## Output files

| File | Contents |
|---|---|
| `data/{split}/{split}_meta.csv` | `trial_id, image_path, source_id, country, cluster, lat, lon, has_text, ocr_text, script, split` |
| `data/build_log.txt` | dataset build log: candidate pools, filter counts, text counts, substitutions, shortfalls |
| `outputs/{,dev/}raw_outputs.jsonl` | one record per trial (SPEC 5.5): model/revision/prompt versions, temperature, candidate order, logprobs, probs, `n_answer_tokens`, top-3 with rationales, `max_pixels`, `seed`, runtimes |
| `outputs/study_dataset.csv` | **the submitted dataset** (SPEC 6 columns + `temperature`, `max_pixels`); `prompt_version` looks like `rank=v1;explain=v1` |
| `outputs/dev/dev_dataset.csv` | the same thing for dev (also the source of the study app's practice trials) |
| `outputs/temperature.json` | `T` fitted on dev with NLL before and after; used by `predict.py` and `server.py` when present |
| `outputs/dev/baseline_json{.jsonl,_summary.json}` | JSON baseline outputs and summary (SPEC 5.4) |
| `outputs/annotations.csv` | `trial_id, rater, error_category, rationale_faithful_top1, human_catchable, note, timestamp`; OVR is derived (`ai_p1 >= 0.6` and wrong), not annotated |
| `outputs/report/` | `metrics.json`, `summary.md`, `error_table.md`, `reliability.png`, `confusion_matrix.png`, `per_country_accuracy.png` |
| `study_app/responses.db` | Assignment 3 responses and surveys (SPEC 11.4) |

## Results

Placeholders. Fill these in from `outputs/report/metrics.json` once the study run is final.

| Metric (study, n=200) | Value | 95% CI |
|---|---|---|
| Top-1 accuracy | _[TODO]_ | _[TODO]_ |
| Top-3 accuracy | _[TODO]_ | _[TODO]_ |
| MRR | _[TODO]_ | _[TODO]_ |
| ECE (10 bins) | _[TODO]_ | — |
| Human-catchable share of top-1 errors | _[TODO]_ | — |
| Fitted T (dev) | _[TODO]_ | — |

## Tests

```bash
python -m pytest                       # fast suite; real-model and network tests are skipped
RUN_MODEL_TESTS=1 python -m pytest     # also loads the real VLM (slow; server/CSV parity on 5 cached images)
RUN_NETWORK_TESTS=1 python -m pytest   # also downloads from Hugging Face
```

## Data, model and licenses

- **Images:** [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (`osv5m/osv5m`, test split),
  derived from [Mapillary](https://www.mapillary.com) street-level imagery and released under **CC-BY-SA 4.0**.
  Every row keeps its original `source_id` for attribution. Images in `data/` are redistributed under the same
  license, and faces and plates are blurred by Mapillary.
- **Model:** [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), **Apache-2.0**,
  pinned at revision `89644892e4d85e24eaac8bacfd4f463576704203`. The model ID and revision are logged in every
  output record and CSV row.
- **Code:** written for UMich CSE594 coursework.

## References

- Astruc, G., Dufour, N., Siglidis, I., Aronssohn, C., Bouia, N., Fu, S., Loiseau, R., Nguyen, V. N., Raude, C.,
  Vincent, E., Xu, L., Zhou, H., & Landrieu, L. (2024). OpenStreetView-5M: The Many Roads to Global Visual
  Geolocation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.
  arXiv:2404.18873.
- Hays, J., & Efros, A. A. (2008). IM2GPS: Estimating geographic information from a single image. In
  *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. In
  *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1321–1330.
- Qwen Team (2025). *Qwen3-VL-2B-Instruct* [model card]. Hugging Face, `Qwen/Qwen3-VL-2B-Instruct`, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`. https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Bansal, G., Wu, T., Zhou, J., Fok, R., Nushi, B., Kamar, E., Ribeiro, M. T., & Weld, D. (2021). Does the Whole
  Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance. In *Proceedings of the 2021
  CHI Conference on Human Factors in Computing Systems*. https://doi.org/10.1145/3411764.3445717
