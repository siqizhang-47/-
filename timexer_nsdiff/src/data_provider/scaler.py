"""Per-column standardisation (design document, 5.1 / 5.2 / 24.2 / 24.10).

Three rules are enforced by construction:

* targets and weather use *different* scaler objects;
* every column gets its own mean/std -- never a shared statistic;
* statistics are fitted on the training rows only.

Binary flags (IsWeekend, IsHoliday) and the cyclic calendar encodings never
reach a scaler: they are produced directly in [0, 1] / [-1, 1].
"""
from __future__ import annotations

import numpy as np


class ColumnStandardScaler:
    def __init__(self, name: str = "scaler", eps: float = 1e-8):
        self.name = name
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> "ColumnStandardScaler":
        """x: [T, C] -- statistics are computed independently per column."""
        self.mean_ = x.mean(axis=0).astype(np.float32)
        self.std_ = x.std(axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Works for any trailing-dim-C array (numpy or torch)."""
        if hasattr(x, "device"):  # torch tensor
            import torch
            mean = torch.as_tensor(self.mean_, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(self.std_, device=x.device, dtype=x.dtype)
            return x * std + mean
        return x * self.std_ + self.mean_

    def inverse_scale_only(self, x):
        """Rescale a *spread* (std / interval width): no mean shift."""
        if hasattr(x, "device"):
            import torch
            std = torch.as_tensor(self.std_, device=x.device, dtype=x.dtype)
            return x * std
        return x * self.std_

    def state_dict(self):
        return {"name": self.name, "mean": self.mean_, "std": self.std_}

    def load_state_dict(self, state):
        self.name = state["name"]
        self.mean_ = np.asarray(state["mean"], dtype=np.float32)
        self.std_ = np.asarray(state["std"], dtype=np.float32)
        return self
