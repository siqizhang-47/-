"""Sliding-window start indices per split, with strict rules (spec 3.10):

* history [i, i+L) and future [i+L, i+L+H) must lie inside the same split;
* any NaN in history target / history weather / future weather / future target
  drops the whole window.
"""
import numpy as np


def valid_window_starts(
    split_start: int,
    split_end: int,
    context_length: int,
    prediction_length: int,
    target_finite: np.ndarray,
    weather_finite: np.ndarray,
):
    """Return (starts_all, starts_valid) for one split.

    target_finite / weather_finite: [N] bool — True when EVERY channel of the
    row is finite.
    """
    L, H = context_length, prediction_length
    total = L + H
    first = split_start
    last = split_end - total  # inclusive last start
    if last < first:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts_all = np.arange(first, last + 1, dtype=np.int64)

    row_ok = target_finite & weather_finite
    # window valid iff all rows in [i, i+total) are ok -> rolling minimum
    ok = row_ok.astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(ok)])
    window_sum = csum[starts_all + total] - csum[starts_all]
    starts_valid = starts_all[window_sum == total]
    return starts_all, starts_valid


def build_split_indices(
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
    context_length: int,
    prediction_length: int,
    target_finite: np.ndarray,
    weather_finite: np.ndarray,
):
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }
    out = {"train_end": train_end, "val_end": val_end}
    for name, (s, e) in bounds.items():
        starts_all, starts_valid = valid_window_starts(
            s, e, context_length, prediction_length, target_finite, weather_finite
        )
        out[f"{name}_starts_all"] = starts_all
        out[f"{name}_starts"] = starts_valid
    return out
