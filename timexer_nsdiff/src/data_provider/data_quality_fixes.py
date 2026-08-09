"""Mandatory data-quality repairs for the HEEW dataset (design document, 2.1).

Three defects are fixed here, in this order:

1. GHI is stamped in UTC while every other column is local MST (UTC-7).  The
   column is shifted by -7 hours and the last 7 rows (whose GHI becomes NaN)
   are dropped.
2. 2016-02-29 and 2020-02-29 were deleted from the source table, leaving two
   24-hour gaps.  They are *not* filled -- the sampler must instead refuse any
   window that straddles them, so this module only reports the gap positions.
3. A handful of weather zeros are missing-value fillers (0 degF air temperature
   and 0 %% relative humidity are impossible in Tucson).  They are flagged as
   NaN and repaired by time-linear interpolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLS = ["Electricity", "PV", "Cooling", "Heat"]
WEATHER_RAW_COLS = ["Temperature", "Dew Point", "Humidity", "GHI"]


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble a local-time (MST, no DST) timestamp column from Year/Month/Day/Hour."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Hour"])
    )
    return df


def fix_ghi_timezone(df: pd.DataFrame, shift_hours: int = 7, verbose: bool = True) -> pd.DataFrame:
    """Shift GHI back by `shift_hours` and drop the tail rows that lose their value."""
    df = df.copy()
    if verbose:
        before = df["GHI"].corr(df["PV"])
    df["GHI"] = df["GHI"].shift(-shift_hours)
    df = df.iloc[:-shift_hours].reset_index(drop=True)
    if verbose:
        after = df["GHI"].corr(df["PV"])
        print(f"[data-fix] GHI shifted by -{shift_hours}h | corr(GHI, PV): "
              f"{before:+.3f} -> {after:+.3f} | dropped {shift_hours} tail rows")
    return df


def fix_weather_zero_fills(df: pd.DataFrame, dew_point_jump: float = 10.0,
                           verbose: bool = True) -> pd.DataFrame:
    """Mark impossible weather zeros as missing and interpolate them in time.

    Temperature == 0 degF and Humidity == 0 %% are always treated as fillers.
    A Dew Point of exactly 0 degF is physically possible in winter, so it is only
    flagged when it is an isolated jump: the mean of the nearest non-zero
    neighbours differs from 0 by more than `dew_point_jump` degF.
    """
    df = df.copy()
    n_flagged = {}

    for col in ["Temperature", "Humidity"]:
        mask = df[col] == 0
        n_flagged[col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    dp = df["Dew Point"]
    zero_mask = dp == 0
    non_zero = dp.where(~zero_mask)
    neighbour = (non_zero.ffill() + non_zero.bfill()) / 2.0
    jump_mask = zero_mask & (neighbour.abs() > dew_point_jump)
    n_flagged["Dew Point"] = int(jump_mask.sum())
    df.loc[jump_mask, "Dew Point"] = np.nan

    df = df.set_index("timestamp")
    df[WEATHER_RAW_COLS] = df[WEATHER_RAW_COLS].interpolate(method="time", limit_direction="both")
    df = df.reset_index()

    if verbose:
        total = sum(n_flagged.values())
        print(f"[data-fix] zero-fill repaired: {n_flagged} "
              f"({100.0 * total / (len(df) * len(WEATHER_RAW_COLS)):.3f}% of weather cells)")
    return df


def find_time_gaps(timestamps: pd.Series, freq_hours: int = 1) -> list:
    """Return [(index_before_gap, gap_length_in_hours)] for every discontinuity."""
    delta = timestamps.diff().dt.total_seconds() / 3600.0
    gap_idx = np.where(delta.values[1:] != freq_hours)[0]
    return [(int(i), float(delta.values[i + 1])) for i in gap_idx]


def contiguous_window_mask(timestamps: pd.Series, window: int, horizon: int,
                           freq_hours: int = 1, verbose: bool = True) -> np.ndarray:
    """Boolean mask over window start indices t0.

    ``mask[t0]`` is True iff rows ``t0 ... t0+window+horizon-1`` are strictly
    consecutive in time, i.e. the sample never straddles a leap-day gap.
    """
    ts = timestamps.values.astype("datetime64[h]").astype(np.int64)
    total = window + horizon
    n_valid = len(ts) - total + 1
    if n_valid <= 0:
        return np.zeros(0, dtype=bool)
    starts = ts[:n_valid]
    ends = ts[total - 1:total - 1 + n_valid]
    mask = (ends - starts) == (total - 1) * freq_hours
    if verbose:
        print(f"[data-fix] contiguous windows: {int(mask.sum())}/{n_valid} "
              f"({n_valid - int(mask.sum())} dropped for crossing a time gap)")
    return mask


def leap_corrected_dayofyear(timestamps: pd.Series) -> np.ndarray:
    """Day-of-year with 29 Feb removed, so 1 March is DoY=60 in every year."""
    doy = timestamps.dt.dayofyear
    leap_after_feb = timestamps.dt.is_leap_year & (timestamps.dt.month >= 3)
    return (doy - leap_after_feb.astype(int)).to_numpy()
