"""Phase 0 acceptance: rows / column order / date boundaries / cleaning."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/processed/energy.csv"

pytestmark = pytest.mark.skipif(
    not CSV.exists(), reason="run scripts/prepare_data.py first")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV, parse_dates=["date"])


def test_shape_and_columns(df):
    assert len(df) == 78888
    assert list(df.columns) == ["date", "PV", "Electricity", "Cooling", "Heat",
                                "Temperature", "DewPoint", "Humidity", "GHI"]


def test_date_boundaries(df):
    assert df["date"].iloc[0] == pd.Timestamp("2014-01-01 00:00")
    assert df["date"].iloc[43823] == pd.Timestamp("2018-12-31 23:00")   # train end
    assert df["date"].iloc[43824] == pd.Timestamp("2019-01-01 00:00")   # val start
    assert df["date"].iloc[52583] == pd.Timestamp("2019-12-31 23:00")   # val end
    assert df["date"].iloc[52584] == pd.Timestamp("2020-01-01 00:00")   # deploy start
    assert df["date"].iloc[-1] == pd.Timestamp("2022-12-31 23:00")      # deploy end


def test_no_nan_no_zero_markers(df):
    assert not df.drop(columns=["date"]).isna().any().any()
    # anomalous zeros (missing markers) must be gone; the handful of genuine
    # dew-point zero crossings (neighbors also near 0) are allowed to remain
    for col in ("DewPoint", "Humidity", "Temperature"):
        vals = df[col].values
        zero_idx = (vals == 0).nonzero()[0]
        for i in zero_idx:
            neighbors = vals[max(i - 2, 0):i + 3]
            assert abs(neighbors[neighbors != 0].mean()) <= 8, \
                f"{col} still has an anomalous zero marker at row {i}"
    assert (df["Humidity"] == 0).sum() == 0      # 0% humidity is never genuine
    assert (df["Temperature"] == 0).sum() == 0   # 0degF amid 50-70degF is never genuine


def test_dataset_split_boundaries(df):
    from backbone.data_loader import Dataset_MTS

    for status, first_row in [("train", 0), ("val", 43824 - 48), ("test", 52584 - 48)]:
        ds = Dataset_MTS("energy", str(CSV), cont_len=48, pred_len=24, status=status)
        assert ds.dates[0] == df["date"].iloc[first_row]
    ds = Dataset_MTS("energy", str(CSV), cont_len=48, pred_len=24, status="test")
    assert ds.dates[-1] == pd.Timestamp("2022-12-31 23:00")


def test_scalers_fit_on_train_only(df):
    from backbone.data_loader import Dataset_MTS, TARGET_COLS

    ds = Dataset_MTS("energy", str(CSV), cont_len=48, pred_len=24, status="test")
    train_mean = df[TARGET_COLS].values[:43824].mean(axis=0)
    assert abs(ds.scaler_target.mean_ - train_mean).max() < 1e-6
