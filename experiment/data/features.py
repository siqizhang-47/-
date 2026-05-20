"""Feature construction for EWELD CSV files.

Each U*.csv is a 15-min sampled univariate series with weather covariates and
20 extreme-weather event labels. We keep labels 01-12 and drop 13-20 per the
experiment spec, drop the textual columns 'Wind' / 'Condition' and the 'User'
column, and add cyclic time features.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


WEATHER_COLS: List[str] = [
    "Temperature",
    "Dew Point",
    "Humidity",
    "Wind Speed",
    "Wind Gust",
    "Pressure",
]

TIME_FEATURES: List[str] = [
    "slot_sin",
    "slot_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]

EVENT_LABELS_USED: List[str] = [f"e{idx:02d}" for idx in range(1, 13)]
EVENT_LABELS_DROPPED: List[str] = [f"e{idx:02d}" for idx in range(13, 21)]

EVENT_FAMILIES = {
    "temp": [f"e{idx:02d}" for idx in range(1, 5)],
    "wind": [f"e{idx:02d}" for idx in range(5, 8)],
    "typhoon": [f"e{idx:02d}" for idx in range(8, 13)],
}

LOAD_COL = "Load"


def _rename_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename messy ``NNxxxx`` event columns to compact ``eNN`` form."""
    mapping = {}
    for col in df.columns:
        if len(col) >= 2 and col[:2].isdigit():
            idx = int(col[:2])
            if 1 <= idx <= 20:
                mapping[col] = f"e{idx:02d}"
    return df.rename(columns=mapping)


def _add_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    t = pd.to_datetime(df[time_col])
    slot = t.dt.hour * 4 + t.dt.minute // 15  # 0..95
    dow = t.dt.dayofweek                       # 0..6
    month = t.dt.month - 1                     # 0..11
    df = df.copy()
    df["slot_sin"] = np.sin(2 * math.pi * slot / 96.0)
    df["slot_cos"] = np.cos(2 * math.pi * slot / 96.0)
    df["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * math.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * math.pi * month / 12.0)
    df["is_weekend"] = (dow >= 5).astype(np.float32)
    return df


def build_user_dataframe(csv_path: str | Path, freq: str = "15min") -> Optional[pd.DataFrame]:
    """Load a single U*.csv and produce the cleaned feature table.

    Returns a DataFrame indexed by an evenly-spaced 15-min timestamp with
    columns:

        Load, Temperature, Dew Point, Humidity, Wind Speed, Wind Gust,
        Pressure, slot_sin, slot_cos, dow_sin, dow_cos, month_sin, month_cos,
        is_weekend, e01..e12, miss_mask

    'miss_mask' is 1 where Load was originally missing (used for quality
    filtering). Returns ``None`` on unrecoverable load errors.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "Time" not in df.columns or LOAD_COL not in df.columns:
        return None

    df = _rename_event_columns(df)

    drop_cols = [c for c in ("User", "Wind", "Condition") if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.drop(columns=[c for c in EVENT_LABELS_DROPPED if c in df.columns], errors="ignore")

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").drop_duplicates(subset=["Time"])
    df = df.set_index("Time")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_index)
    df.index.name = "Time"

    df["miss_mask"] = df[LOAD_COL].isna().astype(np.float32)

    for col in WEATHER_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in EVENT_LABELS_USED:
        if col not in df.columns:
            df[col] = 0.0
    df[LOAD_COL] = pd.to_numeric(df[LOAD_COL], errors="coerce").astype(np.float32)

    # Fill interior gaps with time-interpolation, then catch edge-of-series gaps
    # with ffill/bfill. Columns that are entirely NaN for this user (the raw CSV
    # simply did not carry them) stay NaN here on purpose so the city-level
    # statistics computed downstream with nanmean/nanstd naturally ignore them;
    # the residual NaNs are zeroed out *after* z-score in normalize.transform.
    df[WEATHER_COLS] = (
        df[WEATHER_COLS]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    df[LOAD_COL] = (
        df[LOAD_COL]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    df[EVENT_LABELS_USED] = df[EVENT_LABELS_USED].fillna(0.0).astype(np.float32)

    df = df.reset_index()
    df = _add_time_features(df, time_col="Time")
    df = df.set_index("Time")

    keep = [LOAD_COL] + WEATHER_COLS + TIME_FEATURES + EVENT_LABELS_USED + ["miss_mask"]
    df = df[keep]
    return df
