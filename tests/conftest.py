import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "model" in item.keywords and not os.environ.get("RUN_MODEL_TESTS"):
            item.add_marker(pytest.mark.skip(reason="set RUN_MODEL_TESTS=1"))
        if "network" in item.keywords and not os.environ.get("RUN_NETWORK_TESTS"):
            item.add_marker(pytest.mark.skip(reason="set RUN_NETWORK_TESTS=1"))
