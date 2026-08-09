"""HEEW window dataset (design document, 2 / 4 / 5 / 19).

One sample is

    history_energy   [L, 4]   past L hours of Electricity/PV/Cooling/Heat (standardised)
    future_calendar  [H, 7]   raw calendar of the next H hours, encoded to 11 dims in the model
    future_weather   [H, 4]   Temperature, Dew Point, Humidity, GHI(shifted) (standardised)
    future_energy    [H, 4]   labels (standardised)
    group_info       [H, 3]   hour, IsWeekend, raw Temperature -- used only by the conditional metrics

The 7 raw calendar channels are ``[Year, Month, DayOfYear, Hour, Weekday,
IsWeekend, IsHoliday]``.  Note that day-of-month is replaced by (leap-corrected)
day-of-year, because section 5.3 requires the DoY cyclic encoding and
day-of-month has no physical meaning for load.

Splits are assigned by the year of the *first predicted hour*, so the first 2022
test window may legitimately read history from the end of 2021 (rolling
evaluation, section 4).  Scaler statistics are fitted on training rows only.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_quality_fixes import (TARGET_COLS, WEATHER_RAW_COLS, build_timestamp,
                                 contiguous_window_mask, find_time_gaps, fix_ghi_timezone,
                                 fix_weather_zero_fills, leap_corrected_dayofyear)
from .holiday_features import HolidayFeatureBuilder
from .scaler import ColumnStandardScaler

WEATHER_COLS = list(WEATHER_RAW_COLS)
CALENDAR_RAW_COLS = ["Year", "Month", "DayOfYear", "Hour", "Weekday", "IsWeekend", "IsHoliday"]

# Fixed exogenous token order (design document, section 22) -- 15 tokens:
# 11 calendar channels + 4 measured weather variables.
EXO_TOKEN_NAMES = [
    "YearTrend", "MonthSin", "MonthCos", "DoYSin", "DoYCos", "HourSin", "HourCos",
    "WeekdaySin", "WeekdayCos", "IsWeekend", "IsHoliday",
    "Temperature", "DewPoint", "Humidity", "GHI",
]
# 0 = calendar (deterministic), 1 = weather (needs a forecast)
EXO_TOKEN_TYPES = [0] * 11 + [1, 1, 1, 1]


@dataclass
class SplitYears:
    train: tuple = (2014, 2020)
    val: tuple = (2021, 2021)
    test: tuple = (2022, 2022)


# --------------------------------------------------------------------------- raw table
def load_heew_table(xlsx_path: str, sheet: str = "Aligned_Data", verbose: bool = True) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df = build_timestamp(df)
    if verbose:
        print(f"[data] loaded {df.shape[0]} rows from {os.path.basename(xlsx_path)}::{sheet}")

    df = fix_ghi_timezone(df, verbose=verbose)
    df = fix_weather_zero_fills(df, verbose=verbose)

    gaps = find_time_gaps(df["timestamp"])
    if verbose:
        for i, hours in gaps:
            print(f"[data] time gap after {df['timestamp'].iloc[i]}: {hours:.0f}h "
                  f"(leap day removed from the source table)")

    hb = HolidayFeatureBuilder(verbose=verbose)
    df["DayOfYear"] = leap_corrected_dayofyear(df["timestamp"])
    df["IsWeekend"] = hb.is_weekend(df["Weekday"].to_numpy())
    df["IsHoliday"] = hb.is_holiday(df["timestamp"])
    return df


# --------------------------------------------------------------------------- bundle
class HEEWData:
    """Holds the full preprocessed arrays plus the per-split window indices."""

    def __init__(self, xlsx_path: str, window: int = 168, horizon: int = 24,
                 splits: SplitYears = SplitYears(), cache_dir: str | None = None,
                 verbose: bool = True):
        self.window = window
        self.horizon = horizon
        self.splits = splits
        self.verbose = verbose

        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            key = hashlib.md5(
                f"{os.path.abspath(xlsx_path)}|{os.path.getmtime(xlsx_path)}".encode()
            ).hexdigest()[:10]
            cache_path = os.path.join(cache_dir, f"heew_{key}.npz")

        if cache_path is not None and os.path.exists(cache_path):
            blob = np.load(cache_path, allow_pickle=True)
            self.targets = blob["targets"]
            self.calendar = blob["calendar"]
            self.weather = blob["weather"]
            self.timestamps = pd.to_datetime(blob["timestamps"])
            if verbose:
                print(f"[data] restored preprocessed arrays from {cache_path}")
        else:
            df = load_heew_table(xlsx_path, verbose=verbose)
            self.targets = df[TARGET_COLS].to_numpy(dtype=np.float32)
            self.calendar = df[CALENDAR_RAW_COLS].to_numpy(dtype=np.float32)
            self.weather = df[WEATHER_COLS].to_numpy(dtype=np.float32)
            self.timestamps = df["timestamp"]
            if cache_path is not None:
                np.savez_compressed(cache_path, targets=self.targets, calendar=self.calendar,
                                    weather=self.weather,
                                    timestamps=self.timestamps.to_numpy().astype("datetime64[h]"))
                if verbose:
                    print(f"[data] cached preprocessed arrays to {cache_path}")

        self.timestamps = pd.Series(pd.to_datetime(self.timestamps)).reset_index(drop=True)
        self._build_windows()
        self._fit_scalers()

    # ----------------------------------------------------------------- windows
    def _build_windows(self):
        mask = contiguous_window_mask(self.timestamps, self.window, self.horizon,
                                      verbose=self.verbose)
        starts = np.nonzero(mask)[0]
        # split is decided by the year of the first predicted hour
        first_future_year = self.timestamps.dt.year.to_numpy()[starts + self.window]

        def pick(rng):
            lo, hi = rng
            return starts[(first_future_year >= lo) & (first_future_year <= hi)]

        self.window_index = {
            "train": pick(self.splits.train),
            "val": pick(self.splits.val),
            "test": pick(self.splits.test),
        }
        # Rows usable for fitting scalers: everything whose timestamp is in the training years.
        year = self.timestamps.dt.year.to_numpy()
        self.train_row_mask = (year >= self.splits.train[0]) & (year <= self.splits.train[1])
        if self.verbose:
            print(f"[data] windows  train={len(self.window_index['train'])} "
                  f"val={len(self.window_index['val'])} test={len(self.window_index['test'])}")

    # ----------------------------------------------------------------- scalers
    def _fit_scalers(self):
        self.target_scaler = ColumnStandardScaler("target").fit(self.targets[self.train_row_mask])
        self.weather_scaler = ColumnStandardScaler("weather").fit(self.weather[self.train_row_mask])
        self.targets_scaled = self.target_scaler.transform(self.targets)
        self.weather_scaled = self.weather_scaler.transform(self.weather)
        if self.verbose:
            for name, sc in [("target", self.target_scaler), ("weather", self.weather_scaler)]:
                print(f"[data] {name} scaler mean={np.round(sc.mean_, 3).tolist()} "
                      f"std={np.round(sc.std_, 3).tolist()}")

    def dataset(self, split: str) -> "HEEWWindowDataset":
        return HEEWWindowDataset(self, split)


class HEEWWindowDataset(Dataset):
    def __init__(self, data: HEEWData, split: str):
        self.d = data
        self.split = split
        self.starts = data.window_index[split]
        self.L = data.window
        self.H = data.horizon
        # raw (unscaled) temperature column index, for the conditional metric groups
        self.temp_col = WEATHER_COLS.index("Temperature")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t0 = int(self.starts[i])
        hs, he = t0, t0 + self.L
        fs, fe = he, he + self.H

        history_energy = self.d.targets_scaled[hs:he]
        future_energy = self.d.targets_scaled[fs:fe]
        future_calendar = self.d.calendar[fs:fe]
        future_weather = self.d.weather_scaled[fs:fe]

        hour = self.d.calendar[fs:fe, CALENDAR_RAW_COLS.index("Hour")]
        is_weekend = self.d.calendar[fs:fe, CALENDAR_RAW_COLS.index("IsWeekend")]
        temp_raw = self.d.weather[fs:fe, self.temp_col]
        group_info = np.stack([hour, is_weekend, temp_raw], axis=-1).astype(np.float32)

        return {
            "history_energy": torch.from_numpy(np.ascontiguousarray(history_energy)),
            "future_energy": torch.from_numpy(np.ascontiguousarray(future_energy)),
            "future_calendar": torch.from_numpy(np.ascontiguousarray(future_calendar)),
            "future_weather": torch.from_numpy(np.ascontiguousarray(future_weather)),
            "group_info": torch.from_numpy(group_info),
            "start_index": torch.tensor(t0, dtype=torch.long),
        }
