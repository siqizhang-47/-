import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.seed import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    """The single global seed, applied once per test for determinism."""
    set_seed()
