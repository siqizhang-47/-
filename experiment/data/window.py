"""Sliding-window construction and pooled multi-user dataset.

For each user, generate sliding windows independently within each split.
Train stride = 4 (one sample per hour), val/test stride = 1 (every 15 min).

The Torch ``Dataset`` returned by :func:`build_pooled_dataset` yields tuples
``(x, x_mark, y, y_mark, event_pred, user_idx)`` where ``x`` already has the
target column (Load) placed at the **last** channel — this matches the
``features='MS'`` convention used by the iTransformer / TimeMixer / TimeXer
baselines (target dimension = -1 in the loss).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    EVENT_LABELS_USED,
    LOAD_COL,
    TIME_FEATURES,
    WEATHER_COLS,
)


@dataclass
class WindowConfig:
    seq_len: int = 96
    pred_len: int = 96
    train_stride: int = 4
    eval_stride: int = 1
    feature_set: str = "load_weather_event"  # one of load / load_weather / load_weather_event


def feature_columns(feature_set: str) -> List[str]:
    """Channel layout. Load is always placed LAST (target column for MS)."""
    if feature_set == "load":
        cols = list(TIME_FEATURES)
    elif feature_set == "load_weather":
        cols = list(WEATHER_COLS) + list(TIME_FEATURES)
    elif feature_set == "load_weather_event":
        cols = list(WEATHER_COLS) + list(TIME_FEATURES) + list(EVENT_LABELS_USED)
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return cols + [LOAD_COL]


class WindowDataset(Dataset):
    """Sliding-window dataset over a single contiguous block per user."""

    def __init__(
        self,
        x: np.ndarray,
        events_raw: np.ndarray,
        time_marks: np.ndarray,
        user_idx: int,
        seq_len: int,
        pred_len: int,
        stride: int,
    ) -> None:
        self.x = x.astype(np.float32)
        self.events_raw = events_raw.astype(np.float32)
        self.time_marks = time_marks.astype(np.float32)
        self.user_idx = user_idx
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        n = x.shape[0]
        last = n - seq_len - pred_len
        if last < 0:
            self.starts = np.empty(0, dtype=np.int64)
        else:
            self.starts = np.arange(0, last + 1, stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        L, H = self.seq_len, self.pred_len
        x_enc = self.x[s : s + L]
        x_mark = self.time_marks[s : s + L]
        y = self.x[s + L : s + L + H]
        y_mark = self.time_marks[s + L : s + L + H]
        ev_pred = self.events_raw[s + L : s + L + H]
        return (
            torch.from_numpy(x_enc),
            torch.from_numpy(x_mark),
            torch.from_numpy(y),
            torch.from_numpy(y_mark),
            torch.from_numpy(ev_pred),
            torch.tensor(self.user_idx, dtype=torch.long),
        )


def _build_time_marks(index: pd.DatetimeIndex) -> np.ndarray:
    """5-d time marks used by Time-Series-Library ``timeF`` embedding with freq='t'.

    Returns ``[month, day, weekday, hour, minute]`` normalized to ``[-0.5, 0.5]``.
    """
    month = (index.month - 1) / 11.0 - 0.5
    day = (index.day - 1) / 30.0 - 0.5
    weekday = index.dayofweek / 6.0 - 0.5
    hour = index.hour / 23.0 - 0.5
    minute = (index.minute // 15) / 3.0 - 0.5
    return np.stack([month, day, weekday, hour, minute], axis=-1).astype(np.float32)


def build_pooled_dataset(
    user_frames: Dict[str, pd.DataFrame],
    splits: Dict[str, Tuple[slice, slice, slice]],
    load_normalizer,
    weather_normalizer,
    user_city: Dict[str, str],
    user_idx_map: Dict[str, int],
    wcfg: WindowConfig,
) -> Tuple[Dict[str, "PooledDataset"], List[str]]:
    """Build pooled train / val / test datasets across many users.

    ``splits[uid]`` is the (train, val, test) slice triple computed by
    :func:`experiment.data.filter.time_splits`.
    """
    cols = feature_columns(wcfg.feature_set)

    train_sets: List[WindowDataset] = []
    val_sets: List[WindowDataset] = []
    test_sets: List[WindowDataset] = []

    for uid, df in user_frames.items():
        city = user_city[uid]
        uidx = user_idx_map[uid]
        load_all = df[LOAD_COL].to_numpy(dtype=np.float32)
        weather_all = df[WEATHER_COLS].to_numpy(dtype=np.float32)
        events_raw = df[EVENT_LABELS_USED].to_numpy(dtype=np.float32)
        time_marks = _build_time_marks(df.index)

        load_norm = load_normalizer.transform(uid, load_all)
        weather_norm = weather_normalizer.transform(city, weather_all)
        time_block = df[TIME_FEATURES].to_numpy(dtype=np.float32)
        event_block = df[EVENT_LABELS_USED].to_numpy(dtype=np.float32)

        feature_blocks: List[np.ndarray] = []
        if wcfg.feature_set == "load":
            feature_blocks = [time_block]
        elif wcfg.feature_set == "load_weather":
            feature_blocks = [weather_norm, time_block]
        elif wcfg.feature_set == "load_weather_event":
            feature_blocks = [weather_norm, time_block, event_block]
        x = np.concatenate(feature_blocks + [load_norm[:, None]], axis=1).astype(np.float32)

        tr, va, te = splits[uid]
        train_sets.append(
            WindowDataset(x[tr], events_raw[tr], time_marks[tr], uidx,
                          wcfg.seq_len, wcfg.pred_len, wcfg.train_stride)
        )
        val_sets.append(
            WindowDataset(x[va], events_raw[va], time_marks[va], uidx,
                          wcfg.seq_len, wcfg.pred_len, wcfg.eval_stride)
        )
        test_sets.append(
            WindowDataset(x[te], events_raw[te], time_marks[te], uidx,
                          wcfg.seq_len, wcfg.pred_len, wcfg.eval_stride)
        )

    pooled = {
        "train": PooledDataset(train_sets),
        "val": PooledDataset(val_sets),
        "test": PooledDataset(test_sets),
    }
    return pooled, cols


class PooledDataset(Dataset):
    """Concatenate per-user :class:`WindowDataset` objects into one dataset."""

    def __init__(self, parts: Sequence[WindowDataset]) -> None:
        self.parts = [p for p in parts if len(p) > 0]
        self.lens = [len(p) for p in self.parts]
        self.cumulative = np.cumsum([0] + self.lens)

    def __len__(self) -> int:
        return int(self.cumulative[-1]) if len(self.lens) > 0 else 0

    def __getitem__(self, idx: int):
        for i, p in enumerate(self.parts):
            start = int(self.cumulative[i])
            end = int(self.cumulative[i + 1])
            if idx < end:
                return p[idx - start]
        raise IndexError(idx)
