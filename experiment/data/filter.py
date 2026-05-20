"""High-quality user filtering per the experiment spec.

Conditions (all must hold):
- effective user data length >= 1 year
- missing ratio < 5%
- train / val / test splits all contain data
- test split has >= 30 1-12 event prediction windows
- non-zero load ratio > 80%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import EVENT_LABELS_USED, LOAD_COL


@dataclass
class FilterConfig:
    min_year: float = 1.0
    max_missing_ratio: float = 0.05
    min_event_windows: int = 30
    min_nonzero_ratio: float = 0.80
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    seq_len: int = 96
    pred_len: int = 96
    samples_per_year: int = 96 * 365


def time_splits(n: int, cfg: FilterConfig) -> Tuple[slice, slice, slice]:
    """Chronological split using train/val/test ratios from ``cfg``.

    NOTE: the source document gives 70% / 10% / 30% which sums to 110%.
    We use ``cfg.train_ratio / val_ratio / test_ratio`` (default 70/10/20)
    and assume the spec contained a typo. Adjust the dataclass if needed.
    """
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def _count_event_windows(events: np.ndarray, pred_len: int) -> int:
    """Count test prediction-windows containing any 1-12 event label = 1."""
    n = events.shape[0]
    if n < pred_len:
        return 0
    has_event_per_step = events.sum(axis=1) > 0
    win = np.lib.stride_tricks.sliding_window_view(has_event_per_step, pred_len)
    return int(win.any(axis=1).sum())


def filter_high_quality_users(
    user_frames: Dict[str, pd.DataFrame],
    cfg: FilterConfig,
) -> Dict[str, pd.DataFrame]:
    kept: Dict[str, pd.DataFrame] = {}
    for uid, df in user_frames.items():
        if df is None or df.empty:
            continue
        n = len(df)
        if n < int(cfg.min_year * cfg.samples_per_year):
            continue
        miss_ratio = float(df["miss_mask"].mean())
        if miss_ratio > cfg.max_missing_ratio:
            continue
        load_arr = df[LOAD_COL].to_numpy(dtype=np.float64)
        finite = np.isfinite(load_arr)
        if not finite.any():
            # Entire load column is NaN/Inf after upstream cleaning — can't fit.
            continue
        nonzero_ratio = float((load_arr[finite] > 0).mean())
        if nonzero_ratio < cfg.min_nonzero_ratio:
            continue
        tr, va, te = time_splits(n, cfg)
        if tr.stop - tr.start < cfg.seq_len + cfg.pred_len:
            continue
        if va.stop - va.start < cfg.seq_len + cfg.pred_len:
            continue
        if te.stop - te.start < cfg.seq_len + cfg.pred_len:
            continue
        events_test = df[EVENT_LABELS_USED].to_numpy()[te]
        n_event_windows = _count_event_windows(events_test, cfg.pred_len)
        if n_event_windows < cfg.min_event_windows:
            continue
        kept[uid] = df
    return kept
