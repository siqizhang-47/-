"""Normalization: per-user z-score on Load, per-city z-score on weather.

Statistics are computed from the training-time region only to avoid leakage.
Event labels (binary) and time features (already bounded) are not scaled.

NaN-safety
~~~~~~~~~~
A user may have small interior gaps (handled upstream by interpolate +
ffill/bfill) but can also be missing a whole weather column. In that case
:func:`fit` uses :func:`np.nanmean` / :func:`np.nanstd` so the column simply
contributes no observations to the city aggregate. If a per-user load slice
ends up entirely NaN the fit falls back to ``mean=0, std=1``. The
``transform`` methods clear any residual NaN with :func:`np.nan_to_num`
(NaN → 0 in z-score space) so downstream tensors never carry NaN into the
model. This eliminates the ``RuntimeWarning: Mean of empty slice`` /
``Degrees of freedom <= 0`` warnings that previously produced ``train=nan``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import LOAD_COL, WEATHER_COLS


def _finite_or(value: np.ndarray | float, fallback: float) -> np.ndarray | float:
    """Replace non-finite entries with ``fallback``.

    Accepts scalars and arrays uniformly.
    """
    if np.isscalar(value):
        return float(value) if np.isfinite(value) else float(fallback)
    arr = np.asarray(value, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, fallback)


@dataclass
class LoadNormalizer:
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    def fit(self, user_id: str, train_load: np.ndarray) -> None:
        m = float(np.nanmean(train_load)) if train_load.size else float("nan")
        s = float(np.nanstd(train_load)) if train_load.size else float("nan")
        if not np.isfinite(m):
            m = 0.0
        if (not np.isfinite(s)) or s < 1e-8:
            s = 1.0
        self.mean[user_id] = m
        self.std[user_id] = s

    def transform(self, user_id: str, x: np.ndarray) -> np.ndarray:
        y = (x - self.mean[user_id]) / self.std[user_id]
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def inverse(self, user_id: str, x: np.ndarray) -> np.ndarray:
        return x * self.std[user_id] + self.mean[user_id]


@dataclass
class WeatherNormalizer:
    mean: Dict[str, np.ndarray] = field(default_factory=dict)
    std: Dict[str, np.ndarray] = field(default_factory=dict)
    columns: List[str] = field(default_factory=lambda: list(WEATHER_COLS))

    def fit(self, city: str, train_blocks: List[np.ndarray]) -> None:
        if not train_blocks:
            m = np.zeros(len(self.columns), dtype=np.float32)
            s = np.ones(len(self.columns), dtype=np.float32)
            self.mean[city] = m
            self.std[city] = s
            return
        stacked = np.concatenate(train_blocks, axis=0)
        valid_per_col = np.sum(np.isfinite(stacked), axis=0)
        # nanmean/nanstd raise "Mean of empty slice" / "Degrees of freedom <= 0"
        # whenever a column is fully NaN — we already detect that case via
        # valid_per_col, so silence the cosmetic warnings.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(stacked, axis=0)
            col_std = np.nanstd(stacked, axis=0)
        m = np.where(valid_per_col > 0, col_mean, 0.0)
        s = np.where(valid_per_col > 1, col_std, 1.0)
        m = np.asarray(_finite_or(m, 0.0), dtype=np.float32)
        s = np.asarray(_finite_or(s, 1.0), dtype=np.float32)
        s = np.where(s < 1e-8, 1.0, s).astype(np.float32)
        self.mean[city] = m
        self.std[city] = s

    def transform(self, city: str, x: np.ndarray) -> np.ndarray:
        y = (x - self.mean[city]) / self.std[city]
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
