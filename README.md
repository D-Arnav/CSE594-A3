# Where was this photo taken? — a human-AI study

Participants see a street-level photo and pick the country it was taken in: United States, Germany, Japan,
South Korea or India. In half of the trials an AI (SmolVLM2-2.2B-Instruct, run locally) shows its top 3 guesses,
each with a confidence and a short rationale; the participant checks that evidence against the photo and decides.

| File | What it does |
|---|---|
| `app.py`, `index.html` | The study: a FastAPI server and one HTML page; answers go to `responses.db` (SQLite) |
| `model.py` | Produces the AI outputs: country ranking, calibrated confidence, rationales |
| `evaluate.py` | Metrics and the error table |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data (stored separately)

The photos come from [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (CC-BY-SA 4.0).
Put the dataset folder at `data/`:

```
data/study/S001.jpg … S200.jpg   200 study photos (40 per country) + study_meta.csv (trial_id, image_path, source_id, country)
data/dev/D001.jpg … D200.jpg     200 dev photos (fit the confidence temperature, practice trials) + dev_meta.csv
data/study_dataset.csv           the study dataset: inputs, ground truth, AI outputs, error labels
data/dev_dataset.csv             the same for the dev photos
data/annotations.csv             error labels from two LLM raters, plus the adjudicated label
```

## Run the study

```bash
uvicorn app:app --port 8501
```

Open http://localhost:8501. Give each participant a link with their own number, `?pid=0`, `?pid=1`, …
(even numbers start without the AI, odd numbers with it); without one, the next free number is used.
To share it from a laptop: `cloudflared tunnel --url http://localhost:8501`. Back up `responses.db`, and restart
the server after changing the data.

## Reproduce

```bash
python evaluate.py                    # metrics + error table, no GPU; also writes the final labels into data/study_dataset.csv
caffeinate -i python model.py dev     # AI outputs for the dev photos + the temperature (~20 min on Apple silicon)
caffeinate -i python model.py study   # AI outputs for the study photos (~20 min)
python model.py S017                  # one trial, printed
```

The model (`HuggingFaceTB/SmolVLM2-2.2B-Instruct` at revision `482adb5`, Apache-2.0, about 4.5 GB) downloads from
Hugging Face on first use. `caffeinate` keeps a Mac awake; leave it out elsewhere.

The full development pipeline (dataset building, tests, earlier iterations) is on the git branch `full-pipeline`.
Read it with `git show full-pipeline:<path>`; checking that branch out and back would delete the untracked `data/`.
