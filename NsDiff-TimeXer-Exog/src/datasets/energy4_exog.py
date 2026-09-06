"""Custom 2018-2019 energy + weather dataset for NsDiff exogenous-estimator experiments.

Only the four target variables are forecast by NsDiff:
    Electricity, PV, Cooling, Heat

The four weather variables are conditioning variables only:
    Temperature, Dew Point, Humidity, GHI

The future weather slice is treated as a known weather forecast.  If actual NWP
forecast columns are available later, replace WEATHER_NAMES / CSV columns without
changing the estimator interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TARGET_NAMES = ["Electricity", "PV", "Cooling", "Heat"]
WEATHER_NAMES = ["Temperature", "Dew Point", "Humidity", "GHI"]
TIME_FEATURE_NAMES = ["hour_of_day", "day_of_week", "day_of_month", "day_of_year"]


class StandardScalerNP:
    """Minimal feature-wise standard scaler that also supports torch tensors."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        return self

    def transform(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        x = np.asarray(x, dtype=np.float32)
        return x * self.std + self.mean


class TargetTransform:
    """Optional physical-domain transform followed by feature-wise standardisation.

    ``pv_transform="log1p"`` maps PV -> log(1 + PV) before scaling (V3 Phase 5,
    plan section 15.1).  Because the model then works in log space, generated
    PV samples are guaranteed to be >= 0 after ``inverse_transform``
    (``expm1``), without any inference-time clamping.
    """

    def __init__(self, pv_transform: str = "none", pv_index: int = 1):
        if pv_transform not in {"none", "log1p"}:
            raise ValueError("pv_transform must be 'none' or 'log1p'")
        self.pv_transform = pv_transform
        self.pv_index = int(pv_index)
        self.scaler = StandardScalerNP()

    # -- physical-domain forward / inverse -------------------------------
    def forward_domain(self, x):
        if self.pv_transform == "none":
            return x
        if torch.is_tensor(x):
            out = x.clone()
            out[..., self.pv_index] = torch.log1p(out[..., self.pv_index].clamp_min(0.0))
            return out
        out = np.array(x, dtype=np.float32, copy=True)
        out[..., self.pv_index] = np.log1p(np.clip(out[..., self.pv_index], 0.0, None))
        return out

    def inverse_domain(self, x):
        if self.pv_transform == "none":
            return x
        if torch.is_tensor(x):
            out = x.clone()
            out[..., self.pv_index] = torch.expm1(out[..., self.pv_index])
            return out
        out = np.array(x, dtype=np.float32, copy=True)
        out[..., self.pv_index] = np.expm1(out[..., self.pv_index])
        return out

    # -- scaler API used by the experiments --------------------------------
    @property
    def mean(self):
        return self.scaler.mean

    @property
    def std(self):
        return self.scaler.std

    def fit(self, x: np.ndarray):
        self.scaler.fit(self.forward_domain(np.asarray(x, dtype=np.float32)))
        return self

    def transform(self, x):
        return self.scaler.transform(self.forward_domain(x))

    def inverse_transform(self, x):
        return self.inverse_domain(self.scaler.inverse_transform(x))


def time_features_hourly(dates: pd.DatetimeIndex) -> np.ndarray:
    """TimeXer/Time-Series-Library style hourly calendar features in [-0.5, 0.5]."""
    return np.stack(
        [
            dates.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5,
            dates.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            (dates.day.to_numpy(dtype=np.float32) - 1.0) / 30.0 - 0.5,
            (dates.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.0 - 0.5,
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class Energy4ExogData:
    root: str = "./dataset"

    num_features: int = 4
    num_weather: int = 4
    num_time_features: int = 4
    freq: str = "h"
    feature_names: Sequence[str] = tuple(TARGET_NAMES)
    weather_names: Sequence[str] = tuple(WEATHER_NAMES)

    def __post_init__(self):
        path = Path(self.root) / "Energy4Exog" / "energy4_exog.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/Energy4Exog/prepare_energy4_exog.py first."
            )
        df = pd.read_csv(path, parse_dates=["date"])
        required = ["date"] + TARGET_NAMES + WEATHER_NAMES
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Energy4Exog CSV missing columns: {missing}")
        df = df[required].copy()
        df = df[(df.date >= "2018-01-01") & (df.date < "2020-01-01")].reset_index(drop=True)
        if len(df) != 17520:
            raise ValueError(f"Expected 17,520 hourly rows for 2018-2019, got {len(df):,}")
        if df[required[1:]].isna().any().any():
            raise ValueError("Missing values found in target/weather columns.")
        delta = df.date.diff().dropna()
        if not (delta == pd.Timedelta(hours=1)).all():
            raise ValueError("Timestamps must be continuous hourly observations.")

        self.df = df
        self.dates = pd.DatetimeIndex(df.date)
        self.targets = df[TARGET_NAMES].to_numpy(dtype=np.float32)
        self.weather = df[WEATHER_NAMES].to_numpy(dtype=np.float32)
        self.time_features = time_features_hourly(self.dates)
        self.length = len(df)


class Energy4ExogWindowDataset(Dataset):
    """Window dataset returning target history/future and exogenous history/future."""

    def __init__(
        self,
        data: Energy4ExogData,
        target_scaler,
        weather_scaler: StandardScalerNP,
        indices: Sequence[int],
        window: int,
        pred_len: int,
        horizon: int = 1,
    ):
        self.data = data
        self.target_scaler = target_scaler
        self.weather_scaler = weather_scaler
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = int(window)
        self.pred_len = int(pred_len)
        self.horizon = int(horizon)

        self.targets_scaled = target_scaler.transform(data.targets).astype(np.float32)
        self.weather_scaled = weather_scaler.transform(data.weather).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        s = int(self.indices[item])
        x0, x1 = s, s + self.window
        y0 = x1 + self.horizon - 1
        y1 = y0 + self.pred_len

        x = torch.from_numpy(self.targets_scaled[x0:x1])
        y = torch.from_numpy(self.targets_scaled[y0:y1])
        ox = torch.from_numpy(self.data.targets[x0:x1])
        oy = torch.from_numpy(self.data.targets[y0:y1])
        tx = torch.from_numpy(self.data.time_features[x0:x1])
        ty = torch.from_numpy(self.data.time_features[y0:y1])
        wx = torch.from_numpy(self.weather_scaled[x0:x1])
        wy = torch.from_numpy(self.weather_scaled[y0:y1])
        return x, y, ox, oy, tx, ty, wx, wy


def _window_starts_for_forecast_interval(
    n: int,
    window: int,
    pred_len: int,
    horizon: int,
    forecast_start_min: int,
    forecast_end_max: int,
):
    """Return history start indices whose forecast lies completely in an interval."""
    starts = []
    first = max(0, forecast_start_min - window - horizon + 1)
    last = n - window - horizon - pred_len + 1
    for s in range(first, last + 1):
        y0 = s + window + horizon - 1
        y1 = y0 + pred_len
        if y0 >= forecast_start_min and y1 <= forecast_end_max:
            starts.append(s)
    return starts


def make_energy4_exog_loaders(
    root: str,
    window: int,
    pred_len: int,
    horizon: int,
    batch_size: int,
    train_ratio: float,
    test_ratio: float,
    num_workers: int,
    shuffle_train: bool = True,
    pv_transform: str = "none",
    max_windows_per_split: int = 0,
):
    data = Energy4ExogData(root=root)
    n = data.length
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test
    train_end = n_train
    val_end = n_train + n_val

    # Model-scale target scaler (optionally log1p on PV) and a purely linear
    # scaler used to report all standardized metrics on one comparable scale.
    target_scaler = TargetTransform(pv_transform=pv_transform, pv_index=TARGET_NAMES.index("PV"))
    target_scaler.fit(data.targets[:train_end])
    data.linear_target_scaler = StandardScalerNP().fit(data.targets[:train_end])
    data.pv_transform = pv_transform
    weather_scaler = StandardScalerNP().fit(data.weather[:train_end])

    train_idx = _window_starts_for_forecast_interval(
        n, window, pred_len, horizon, 0, train_end
    )
    val_idx = _window_starts_for_forecast_interval(
        n, window, pred_len, horizon, train_end, val_end
    )
    test_idx = _window_starts_for_forecast_interval(
        n, window, pred_len, horizon, val_end, n
    )

    if max_windows_per_split and max_windows_per_split > 0:
        # Evenly spaced subset of every split (smoke tests / quick ablations only).
        def _subset(idx):
            if len(idx) <= max_windows_per_split:
                return idx
            pick = np.linspace(0, len(idx) - 1, max_windows_per_split).round().astype(int)
            return [idx[i] for i in pick]
        train_idx, val_idx, test_idx = _subset(train_idx), _subset(val_idx), _subset(test_idx)

    train_ds = Energy4ExogWindowDataset(data, target_scaler, weather_scaler, train_idx, window, pred_len, horizon)
    val_ds = Energy4ExogWindowDataset(data, target_scaler, weather_scaler, val_idx, window, pred_len, horizon)
    test_ds = Energy4ExogWindowDataset(data, target_scaler, weather_scaler, test_idx, window, pred_len, horizon)

    pin = torch.cuda.is_available()
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=shuffle_train, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    print(f"Energy4Exog split rows: train={n_train}, val={n_val}, test={n_test}  (pv_transform={pv_transform})")
    print(f"Energy4Exog windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return data, target_scaler, weather_scaler, train_loader, val_loader, test_loader
