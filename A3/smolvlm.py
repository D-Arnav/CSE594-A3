"""SmolVLM2-2.2B-Instruct suggestions from Assignment 2, precomputed for every study photo by A2's model.py."""
import csv
from pathlib import Path

MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

with open(Path(__file__).parent / "smolvlm_outputs.csv", newline="") as f:
    OUTPUTS = {row["image_id"]: [{"country": row[f"top{i}"], "p": float(row[f"p{i}"]), "rationale": row[f"rationale{i}"]}
                                 for i in (1, 2, 3)]
               for row in csv.DictReader(f)}


def suggest(image_id):
    """-> [{"country", "p", "rationale"}] x 3, ranked, as A2's model produced them."""
    return OUTPUTS[image_id]
