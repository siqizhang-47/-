"""Normalization: per-user z-score on Load, per-city z-score on weather.

Statistics are computed from the training-time region only to avoid leakage.
Event labels (binary) and time features (already bounded) are not scaled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import LOAD_COL, WEATHER_COLS


@dataclass
class LoadNormalizer:
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    def fit(self, user_id: str, train_load: np.ndarray) -> None:
        m = float(np.mean(train_load))
        s = float(np.std(train_load))
        if s < 1e-8:
            s = 1.0
        self.mean[user_id] = m
        self.std[user_id] = s

    def transform(self, user_id: str, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[user_id]) / self.std[user_id]

    def inverse(self, user_id: str, x: np.ndarray) -> np.ndarray:
        return x * self.std[user_id] + self.mean[user_id]


@dataclass
class WeatherNormalizer:
    mean: Dict[str, np.ndarray] = field(default_factory=dict)
    std: Dict[str, np.ndarray] = field(default_factory=dict)
    columns: List[str] = field(default_factory=lambda: list(WEATHER_COLS))

    def fit(self, city: str, train_blocks: List[np.ndarray]) -> None:
        stacked = np.concatenate(train_blocks, axis=0)
        m = np.nanmean(stacked, axis=0)
        s = np.nanstd(stacked, axis=0)
        s = np.where(s < 1e-8, 1.0, s)
        self.mean[city] = m.astype(np.float32)
        self.std[city] = s.astype(np.float32)

    def transform(self, city: str, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[city]) / self.std[city]
