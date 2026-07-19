"""Target / weather transforms (spec section 6).

electricity : z-score with train statistics
cooling/heating/pv : U = log(1 + Y / s_d), s_d = train positive median
weather : z-score with train statistics
All statistics are fit on the TRAIN split valid values only.
"""
import numpy as np
import torch

from src.data.low_carbon_schema import (
    NUM_TARGETS,
    TARGET_NAMES,
    WEATHER_NAMES,
    ZERO_TARGET_INDICES,
)

EPS = 1e-8


class TargetTransform:
    def __init__(self, stats: dict):
        self.stats = stats
        self.elec_mean = float(stats["electricity_mean"])
        self.elec_std = float(stats["electricity_std"])
        # positive scale per target index (index 0 unused)
        self.positive_scale = np.array(stats["positive_scale"], dtype=np.float64)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, target_raw: np.ndarray, observed: np.ndarray, train_end: int):
        """target_raw: [N, 4] raw kW with NaN; observed: [N, 4] bool."""
        stats = {}
        tr = target_raw[:train_end]
        ob = observed[:train_end]
        elec = tr[:, 0][ob[:, 0]]
        stats["electricity_mean"] = float(np.mean(elec))
        stats["electricity_std"] = float(np.std(elec) + EPS)
        scale = [1.0, 1.0, 1.0, 1.0]
        pos_weight = [1.0, 1.0, 1.0]
        zero_rate = {}
        for d in ZERO_TARGET_INDICES:
            vals = tr[:, d][ob[:, d]]
            pos = vals[vals > 0.0]
            if len(pos) == 0:
                raise ValueError(f"no positive train values for target {TARGET_NAMES[d]}")
            scale[d] = float(np.median(pos))
            n_pos = float((vals > 0.0).sum())
            n_zero = float((vals <= 0.0).sum())
            pos_weight[d - 1] = float(n_zero / max(n_pos, 1.0))
            zero_rate[TARGET_NAMES[d]] = float(n_zero / max(len(vals), 1.0))
        stats["positive_scale"] = scale
        stats["gate_pos_weight"] = pos_weight
        stats["train_zero_rate"] = zero_rate
        # raw-space scale used for CRPS normalisation
        stats["train_scale_raw"] = [
            float(np.std(tr[:, d][ob[:, d]]) + EPS) for d in range(NUM_TARGETS)
        ]
        return cls(stats)

    # ------------------------------------------------------------ transform
    def transform(self, y_raw):
        """raw kW -> model space; NaN passes through as NaN. numpy [_, 4]."""
        y = np.array(y_raw, dtype=np.float64, copy=True)
        y[..., 0] = (y[..., 0] - self.elec_mean) / self.elec_std
        for d in ZERO_TARGET_INDICES:
            y[..., d] = np.log1p(np.clip(y[..., d], 0.0, None) / self.positive_scale[d])
        return y.astype(np.float32)

    def inverse(self, u):
        """model space -> raw kW. numpy array with target dim on axis -1 or given axis."""
        y = np.array(u, dtype=np.float64, copy=True)
        return self._inverse_inplace(y).astype(np.float32)

    def _inverse_inplace(self, y):
        y[..., 0] = np.clip(y[..., 0] * self.elec_std + self.elec_mean, 0.0, None)
        for d in ZERO_TARGET_INDICES:
            y[..., d] = self.positive_scale[d] * np.clip(np.expm1(y[..., d]), 0.0, None)
        return y

    # torch versions operating on [..., D, ...] with target dim = `dim`
    def inverse_torch(self, u: torch.Tensor, target_dim: int = -1) -> torch.Tensor:
        u = u.movedim(target_dim, -1)
        out = torch.empty_like(u)
        out[..., 0] = (u[..., 0] * self.elec_std + self.elec_mean).clamp_min(0.0)
        for d in ZERO_TARGET_INDICES:
            out[..., d] = float(self.positive_scale[d]) * torch.expm1(u[..., d]).clamp_min(0.0)
        return out.movedim(-1, target_dim)

    def positive_inverse(self, u, target_index: int):
        """inverse for one zero-inflated target; positive_inverse(0) == 0."""
        s = float(self.positive_scale[target_index])
        if isinstance(u, torch.Tensor):
            return s * torch.expm1(u).clamp_min(0.0)
        return s * np.clip(np.expm1(np.asarray(u, dtype=np.float64)), 0.0, None)


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
