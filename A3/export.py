"""Write every study table to exports/<table>.csv (uses the same env vars as the app)."""
import csv
from contextlib import closing
from pathlib import Path

import db

OUT = Path(__file__).parent / "exports"
TABLES = ("participants", "attention_checks", "trials", "surveys", "ai_shown")

OUT.mkdir(exist_ok=True)
with closing(db.connect()) as conn:
    for table in TABLES:
        cursor = conn.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        with open(OUT / f"{table}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(column[0] for column in cursor.description)
            writer.writerows(rows)
        print(f"{table}: {len(rows)} rows")
