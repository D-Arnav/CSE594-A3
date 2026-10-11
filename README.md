# Where was this photo taken?

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --port 8501           # study app, AI suggestions read from data/study_dataset.csv
LIVE=1 uvicorn app:app --port 8501    # study app, AI suggestions computed live by the model
python model.py                       # regenerate data/study_dataset.csv with the model (~20 min)
python evaluate.py                    # print the report's two tables
```

Open http://localhost:8501/?pid=0, one number per participant. The model downloads on first use (~4.5 GB).
