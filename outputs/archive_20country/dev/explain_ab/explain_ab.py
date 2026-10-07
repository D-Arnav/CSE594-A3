"""A/B explain prompt versions on the first N dev trials (same top-3 as the dev run).
python outputs/dev/explain_ab/explain_ab.py 40 v2"""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from common import load_config
from model import GeoModel
from predict import load_image
import pandas as pd

N, VERSIONS = int(sys.argv[1]), sys.argv[2:]
recs = [json.loads(l) for l in open("outputs/dev/raw_outputs.jsonl")][:N]
meta = pd.read_csv("data/dev/dev_meta.csv").set_index("trial_id")
gm = GeoModel(load_config())
rows = []
for r in recs:
    img = load_image(meta.at[r["trial_id"], "image_path"])
    for k, t in enumerate(r["top3"], 1):
        row = {"trial_id": r["trial_id"], "truth": meta.at[r["trial_id"], "country"], "rank": k,
               "country": t["country"], "v1": t["rationale"]}
        for v in VERSIONS:
            row[v] = gm.explain(img, t["country"], prompt_version=v)
        rows.append(row)
    print(r["trial_id"], flush=True)
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_name("explain_ab.csv"), index=False)
for v in ["v1", *VERSIONS]:
    s = df[v].fillna("")
    print(v, "no-clues by rank:", s.str.contains("No distinctive clues", case=False).groupby(df["rank"]).mean().round(2).tolist(),
          "yes/no opener:", round(s.str.match(r"^(Yes|No)\b").mean(), 2), "mean chars:", round(s.str.len().mean()))
