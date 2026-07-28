"""Target / weather transforms for HEEW.

All three targets are strictly positive continuous loads -> z-score with
TRAIN-split statistics. Weather is z-scored the same way. Statistics are
fit on the TRAIN split valid values only.
"""
import numpy as np
import torch

from src.data.low_carbon_schema import NUM_TARGETS, TARGET_NAMES, WEATHER_NAMES

EPS = 1e-8


class TargetTransform:
    def __init__(self, stats: dict):
        self.stats = stats
        self.mean = np.asarray(stats["target_mean"], dtype=np.float64)
        self.std = np.asarray(stats["target_std"], dtype=np.float64)

    @classmethod
    def fit(cls, target_raw: np.ndarray, observed: np.ndarray, train_end: int):
        tr = target_raw[:train_end]
        ob = observed[:train_end]
        mean, std, scale = [], [], []
        for d in range(NUM_TARGETS):
            vals = tr[:, d][ob[:, d]]
            mean.append(float(np.mean(vals)))
            std.append(float(np.std(vals) + EPS))
        stats = {
            "target_mean": mean,
            "target_std": std,
            # raw-space scale used for CRPS normalisation (validation metric)
            "train_scale_raw": std,
            "target_names": TARGET_NAMES,
        }
        return cls(stats)

    def transform(self, y_raw):
        y = np.asarray(y_raw, dtype=np.float64)
        return ((y - self.mean) / self.std).astype(np.float32)

    def inverse(self, u):
        y = np.asarray(u, dtype=np.float64) * self.std + self.mean
        return np.clip(y, 0.0, None).astype(np.float32)

    def inverse_torch(self, u: torch.Tensor, target_dim: int = -1) -> torch.Tensor:
        u = u.movedim(target_dim, -1)
        mean = torch.as_tensor(self.mean, dtype=u.dtype, device=u.device)
        std = torch.as_tensor(self.std, dtype=u.dtype, device=u.device)
        out = (u * std + mean).clamp_min(0.0)
        return out.movedim(-1, target_dim)


class WeatherTransform:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, weather_raw: np.ndarray, observed: np.ndarray, train_end: int):
        tr = weather_raw[:train_end]
        ob = observed[:train_end]
        mean, std = [], []
        for j in range(tr.shape[1]):
            vals = tr[:, j][ob[:, j]]
            mean.append(float(np.mean(vals)))
            std.append(float(np.std(vals) + EPS))
        return cls(mean, std)

    def transform(self, w_raw):
        w = np.asarray(w_raw, dtype=np.float64)
        return ((w - self.mean) / self.std).astype(np.float32)

    def to_stats(self):
        return {
            "weather_mean": self.mean.tolist(),
            "weather_std": self.std.tolist(),
            "weather_names": WEATHER_NAMES,
        }
