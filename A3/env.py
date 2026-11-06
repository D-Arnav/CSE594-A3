"""Load A3/.env for local runs. Real environment variables win, so hosting needs no .env file."""
import os
from pathlib import Path


def load(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load(Path(__file__).parent / ".env")
