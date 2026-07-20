"""Window dataset returning the batch dict of spec section 5.2."""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.low_carbon_schema import ZERO_TARGET_INDICES
from src.data.target_transform import TargetTransform, WeatherTransform


class LowCarbonWindowDataset(Dataset):
    def __init__(self, artifacts_dir: str, split: str, stride: int = 1):
        assert split in ("train", "val", "test")
        self.split = split
        self.dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "preprocess_stats.json"), encoding="utf-8") as f:
            self.stats = json.load(f)
        self.target_transform = TargetTransform(self.stats)
        self.weather_transform = WeatherTransform(
            self.stats["weather_mean"], self.stats["weather_std"]
        )

        self.target_raw = np.load(os.path.join(artifacts_dir, "target_raw.npy"))
        weather_raw = np.load(os.path.join(artifacts_dir, "weather_raw.npy"))
        self.calendar = np.load(os.path.join(artifacts_dir, "calendar.npy"))
        self.timestamps = np.load(os.path.join(artifacts_dir, "timestamps.npy"))
        self.observed = np.load(os.path.join(artifacts_dir, "observed_mask.npy"))
        split_npz = np.load(os.path.join(artifacts_dir, "split_indices.npz"))
        self.starts = split_npz[f"{split}_starts"][:: max(int(stride), 1)]
        self.L = int(split_npz["context_length"])
        self.H = int(split_npz["prediction_length"])

        # precompute transformed target and condition matrix [N, 11]
        target_model = self.target_transform.transform(self.target_raw)
        # inactive/NaN placeholder 0 in model space (losses use masks, never labels)
        self.target_model = np.nan_to_num(target_model, nan=0.0).astype(np.float32)
        self.target_raw_filled = np.nan_to_num(self.target_raw, nan=0.0).astype(np.float32)
        weather_std = self.weather_transform.transform(weather_raw)
        weather_std = np.nan_to_num(weather_std, nan=0.0).astype(np.float32)
        self.condition = np.concatenate([weather_std, self.calendar], axis=1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        L, H = self.L, self.H
        h_slice = slice(s, s + L)
        f_slice = slice(s + L, s + L + H)
        future_raw = self.target_raw_filled[f_slice]
        future_observed = self.observed[f_slice]
        future_active = future_raw[:, ZERO_TARGET_INDICES] > 0.0
        return {
            "history_target": torch.from_numpy(self.target_model[h_slice]),
            "future_target": torch.from_numpy(self.target_model[f_slice]),
            "history_target_raw": torch.from_numpy(self.target_raw_filled[h_slice]),
            "future_target_raw": torch.from_numpy(future_raw),
            "history_condition": torch.from_numpy(self.condition[h_slice]),
            "future_condition": torch.from_numpy(self.condition[f_slice]),
            "history_observed": torch.from_numpy(self.observed[h_slice]),
            "future_observed": torch.from_numpy(future_observed),
            "future_active": torch.from_numpy(future_active),
            "forecast_start_index": torch.tensor(s + L, dtype=torch.long),
            "timestamps": torch.from_numpy(self.timestamps[f_slice].astype(np.int64)),
        }
