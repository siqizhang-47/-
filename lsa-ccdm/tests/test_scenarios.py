"""Phase 2 acceptance: spot checks on the cached scenarios."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/scenarios/wnorm_on"

pytestmark = pytest.mark.skipif(
    not CACHE.exists() or not list(CACHE.glob("*.npz")),
    reason="run experiments/run_deploy.py first")


@pytest.fixture(scope="module")
def sample_days():
    files = sorted(CACHE.glob("*.npz"))
    step = max(len(files) // 10, 1)
    return [np.load(f) for f in files[::step][:10]]


def test_completeness():
    files = {p.stem for p in CACHE.glob("*.npz")}
    deploy = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    missing = [d for d in deploy if str(d.date()) not in files]
    assert not missing, f"{len(missing)} deployment days missing, e.g. {missing[:3]}"


def test_no_nan_and_magnitude(sample_days):
    for z in sample_days:
        scen, y = z["scenarios"], z["y_true"]
        assert not np.isnan(scen).any() and not np.isnan(y).any()
        # scenario mean has the same order of magnitude as the truth
        mu = np.abs(scen.mean(axis=0)).mean(axis=0) + 1e-6
        yt = np.abs(y).mean(axis=0) + 1e-6
        ratio = mu / yt
        assert (ratio > 0.1).all() and (ratio < 10).all(), ratio


def test_pv_night_below_day(sample_days):
    for z in sample_days:
        scen, ghi = z["scenarios"], z["weather_future"][:, 3]
        pv_mean = scen.mean(axis=0)[:, 0]
        night, day = ghi == 0, ghi > 200
        if night.sum() >= 3 and day.sum() >= 3:
            assert pv_mean[night].mean() < 0.05 * abs(pv_mean[day].mean()) + 1.0
