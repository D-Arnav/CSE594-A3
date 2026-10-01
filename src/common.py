"""Shared constants and helpers used by every script."""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Display name -> ISO 3166-1 alpha-2 (OSV-5M `country` column)
ISO2 = {
    "United States": "US", "Canada": "CA", "Mexico": "MX",
    "Brazil": "BR", "Argentina": "AR", "Chile": "CL",
    "United Kingdom": "GB", "Ireland": "IE",
    "France": "FR", "Spain": "ES", "Italy": "IT", "Germany": "DE",
    "Russia": "RU", "Poland": "PL",
    "Japan": "JP", "South Korea": "KR",
    "India": "IN", "Thailand": "TH",
    "South Africa": "ZA", "Australia": "AU",
    # backups
    "New Zealand": "NZ", "Netherlands": "NL", "Indonesia": "ID", "Turkey": "TR",
}
NAME = {v: k for k, v in ISO2.items()}

CLUSTERS = {
    "North America": ["United States", "Canada", "Mexico"],
    "South America": ["Brazil", "Argentina", "Chile"],
    "British Isles": ["United Kingdom", "Ireland"],
    "Western Europe": ["France", "Spain", "Italy", "Germany"],
    "Eastern Europe": ["Russia", "Poland"],
    "East Asia": ["Japan", "South Korea"],
    "South / Southeast Asia": ["India", "Thailand"],
    "Southern Hemisphere": ["South Africa", "Australia"],
    # backups join the closest cluster if substituted in
    "Backup": ["New Zealand", "Netherlands", "Indonesia", "Turkey"],
}
CLUSTER_OF = {c: k for k, cs in CLUSTERS.items() for c in cs}


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def countries(cfg):
    """Active label space. build_dataset may write data/countries.txt after substitutions."""
    override = DATA / "countries.txt"
    if override.exists():
        return [l.strip() for l in override.read_text().splitlines() if l.strip()]
    return list(cfg["data"]["countries"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
