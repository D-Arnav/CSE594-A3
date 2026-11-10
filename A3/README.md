# Where was this photo taken?

A between-subjects study: participants guess the country of 40 street-level photos, with or without AI suggestions.

| Condition | Link |
|---|---|
| AI assistance | `/?ai-assist=True` |
| No AI | `/?ai-assist=False` |

## Run locally

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo 'GEMINI_API_KEY=...' > .env   # git ignores this file
uvicorn app:app --port 8501
python test_app.py   # checks both conditions with Gemini stubbed; needs no key
```

Without `TURSO_DATABASE_URL`, data goes to a local `study.db`. `.env` can also hold the Turso settings.

## Deploy

1. Push this repository to GitHub.
2. Get a key from [Google AI Studio](https://aistudio.google.com/).
3. Create a [Turso](https://turso.tech/) database, plus its URL and auth token.
4. In [Render](https://render.com/), create a Blueprint from the repository. It reads `render.yaml`. Enter `GEMINI_API_KEY`, `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` when prompted.

The free Render service sleeps after 15 idle minutes and takes about a minute to wake. Its disk is wiped on restart, which is why responses live in Turso.

## Export data

```bash
TURSO_DATABASE_URL=... TURSO_AUTH_TOKEN=... python export.py   # writes exports/*.csv
```

## Design decisions

- **Photos:** each participant gets a random draw of 40 from the 200 A2 study photos, balanced at 8 per country so no country dominates a session. The draw is seeded by participant ID, so a refresh keeps the same trials. `TRIALS_PER_COUNTRY` in `app.py` controls this.
- **Participant ID:** participants enter their U-M uniqname. The ID is uniqname plus condition (`jsmith-ai`, `jsmith-noai`), so one person's two sessions can be paired: `participant_id.split("-")[0]` gives the uniqname, and `created_at` shows which condition came first. Entering the same uniqname on the same link resumes that session. A session ended by attention checks keeps its data, and starting over creates `jsmith-ai-2`. Uniqnames identify people, so remove or replace them before sharing data outside the course.
- **Assignment:** the link sets the condition, and the server stores it.
- **Attention checks:** after photos 13 and 26, the usual photo screen shows a random photo outside the participant's 40, with "Select South Korea." or "Select Germany." in place of the question and no AI suggestions (see `attention.py`). The first miss shows a warning and a reminder. Missing both ends the session, and the participant starts over as a new participant. Answers stay on the server, and the server won't accept the next trial until a due check is answered.
- **AI sources:** in each AI session, a random 20% of trials (8 of 40, `GEMINI_SHARE` in `app.py`) call Gemini live while the participant waits, and the other 80% show the Assignment 2 model's suggestions (SmolVLM2-2.2B-Instruct, precomputed by A2's `model.py`, stored in `smolvlm_outputs.csv`). Every Gemini trial is a new call, never reused across participants. If Gemini fails after retries, that trial shows SmolVLM2's output and is logged as `smolvlm_fallback`, so no one gets stuck. Participants aren't told which model produced a suggestion.
- **Gemini:** `gemini-3.1-flash-lite`, the cheapest free-tier vision model, receives the same photo file participants see. Its probabilities are its own estimates; SmolVLM2's are A2's calibrated softmax (T = 0.59).
- **Response time:** starts when the photo loads (no AI) or when suggestions appear (AI), so API latency isn't counted. Time spent zooming counts.
- **Layout:** the photo gets most of the screen, with answers in a fixed panel and Next always in the same place. A route with one stop per photo shows progress.
- **Zoom:** scroll, pinch, double-click, the on-photo controls, or `+`, `-` and `0` zoom up to 500% around the cursor, and dragging moves the photo (see `static/zoom.js`). Photos are shown unfiltered, so both conditions see the same image.
- **Trusted data:** the server derives correctness and which AI card was chosen. Answers and `images.csv` are never sent to the browser.

## Measurements

| Table | Fields |
|---|---|
| `trials` | answer, correctness, response time, chosen AI rank (0 means an unsuggested country), whether "None of these" was opened, AI source (`smolvlm`, `gemini`, `smolvlm_fallback`), AI top choice and whether it was correct |
| `surveys` | confidence and mental demand (both conditions), geography knowledge, whether the participant already did the other condition, free text, plus trust and usefulness in the AI condition |
| `attention_checks` | each check's answer and whether it passed |
| `ai_shown` | exact suggestions shown on each AI trial, per participant, with their source |

**Valid participants for analysis:** those with a row in `surveys`. A session ended by attention checks never reaches the survey, but its partial data stays for transparency. A participant who missed one check can still finish; `attention_checks` identifies them.

Confidence and mental demand are asked in both conditions, so they can be compared directly, along with response time. Order effects can be read from the data, since uniqname pairing and `created_at` show which condition each person did first. `seen_other_condition` (self-reported) cross-checks it.
