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

## Data

The photos come from [OpenStreetView-5M](https://huggingface.co/datasets/osv5m/osv5m) (CC-BY-SA 4.0).
`data/` is included in the submission but not tracked in git:

```
data/study/S001.jpg … S200.jpg   200 study photos (40 per country) + study_meta.csv (trial_id, image_path, source_id, country)
data/study_dataset.csv           the study dataset: inputs, ground truth, AI outputs, error labels
data/practice/P01.jpg, P02.jpg   2 practice photos (not study trials) + practice.csv with their AI outputs
data/annotations.csv             error labels from two LLM raters, plus the adjudicated label
```

## Run the study

```bash
uvicorn app:app --port 8501
```

Add `LIVE=1` in front to have the AI compute its suggestions live (see below). Open http://localhost:8501. Give each participant a link with their own number, `?pid=0`, `?pid=1`, …
(even numbers start without the AI, odd numbers with it); without one, the next free number is used.
To share it from a laptop: `cloudflared tunnel --url http://localhost:8501`. Back up `responses.db`, and restart
the server after changing the data.

## AI model (`model.py`)

The AI is [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) (Apache-2.0),
pinned to revision `482adb5`. It downloads from Hugging Face on first run (about 4.5 GB; no account or token
needed). For each photo `model.py`:

1. **Ranks the 5 countries:** it asks the model which of the 5 countries (listed in a per-photo random order) the
   photo is from, and scores the log-likelihood of each country name as the answer.
2. **Turns the scores into confidences:** a softmax with temperature T = 0.59, which was fitted beforehand on a
   separate set of calibration photos.
3. **Writes a rationale** for each of the top 3 countries: it asks the model what it sees that suggests that
   country, then cuts the answer to two sentences.

### Run the AI live

```bash
LIVE=1 uvicorn app:app --port 8501    # the study with every AI card computed by the model as the trial is shown
python model.py                       # all 200 study photos -> data/study_dataset.csv (~20 min)
```

The model takes 15–20 s to load, then about 6 s per photo. In the live study the server loads the model at startup. It computes each photo's cards when a participant reaches that trial,
fetching one trial ahead so there is usually no wait, and reuses them for later participants. Without `LIVE=1` the
study reads the same cards from `data/study_dataset.csv`, which needs no GPU. Live and saved study cards are identical on
the machine that made the dataset.

It picks the device automatically (NVIDIA GPU, then Apple silicon, then CPU). A CPU works but is much slower.
Decoding is greedy, so a rerun reproduces the outputs in `data/study_dataset.csv`. On a Mac, prefix long runs with
`caffeinate -i` so the machine doesn't sleep.

## Reproduce the tables

```bash
python evaluate.py      # the report's two tables (metrics, error analysis) from data/study_dataset.csv
                        # and data/annotations.csv; no GPU
```
