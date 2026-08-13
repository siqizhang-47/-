"""Phase 2: daily rolling deployment of the frozen backbone with scenario caching.

For every deployment day (00:00 forecast origin) the frozen CCDM samples M
joint scenarios which are de-normalized (inverse window norm, then inverse
StandardScaler) to physical units and stored as one npz per day:
    scenarios (M, 24, 4) float32, y_true (24, 4), weather_future (24, 4), date

Warmup days from the 2019 validation tail (legal information) are also cached
for the adapters' cold start.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from backbone.data_loader import COV_COLS, TARGET_COLS, data_intervals
from backbone.model import DiffMTS
from utils.time_features import time_features


class FrozenDeployer:
    def __init__(self, cfg, ckpt_path, intervals=None):
        self.cfg = cfg
        self.device = cfg.device
        self.cont_len = cfg.cont_len
        self.pred_len = cfg.pred_len

        df = pd.read_csv(cfg.data_csv, parse_dates=["date"])
        self.dates = pd.DatetimeIndex(df["date"])
        self.target_raw = df[TARGET_COLS].values.astype(np.float64)
        self.cov_raw = df[COV_COLS].values.astype(np.float64)

        intervals = intervals or data_intervals[cfg.data_name]
        train_end = intervals[0]
        self.scaler_target = StandardScaler().fit(self.target_raw[:train_end])
        self.scaler_cov = StandardScaler().fit(self.cov_raw[:train_end])
        self.target = self.scaler_target.transform(self.target_raw).astype(np.float32)
        self.cov = self.scaler_cov.transform(self.cov_raw).astype(np.float32)
        stamp = time_features(self.dates, freq=cfg.freq).transpose(1, 0)
        self.stamp = stamp.astype(np.float32)

        self.model = DiffMTS(cfg)
        self.model.load_weights(ckpt_path)

    def _day_row(self, date):
        idx = self.dates.get_loc(pd.Timestamp(date))
        assert self.dates[idx].hour == 0
        return idx

    @torch.no_grad()
    def sample_day(self, date, M):
        """Generate M de-normalized scenarios for one deployment day."""
        cfg = self.cfg
        i0 = self._day_row(date)
        assert i0 >= self.cont_len, f"not enough history before {date}"
        assert i0 + self.pred_len <= len(self.target)

        hist = np.concatenate([self.target[i0 - self.cont_len:i0],
                               self.cov[i0 - self.cont_len:i0]], axis=-1)
        w_future = self.cov[i0:i0 + self.pred_len]
        x_mark = self.stamp[i0 - self.cont_len:i0]
        y_mark = self.stamp[i0:i0 + self.pred_len]

        to_t = lambda a: torch.from_numpy(a).unsqueeze(0).float().to(self.device)
        x, w = to_t(hist), to_t(w_future)
        x_mark, y_mark = to_t(x_mark), to_t(y_mark)

        norm_stats = None
        if cfg.use_window_norm:
            x, _, mean, std = self.model.instance_normalization(x)
            norm_stats = (mean, std)
        scen = self.model.pred_sampling(x, w, M, x_mark, y_mark)  # (M, pred_len, C)
        if cfg.use_window_norm:
            scen = self.model.instance_denormalization(scen, *norm_stats)
        scen = scen.cpu().numpy()
        # inverse StandardScaler to physical units
        scen = self.scaler_target.inverse_transform(
            scen.reshape(-1, scen.shape[-1])).reshape(scen.shape).astype(np.float32)
        y_true = self.target_raw[i0:i0 + self.pred_len].astype(np.float32)
        weather_future = self.cov_raw[i0:i0 + self.pred_len].astype(np.float32)
        return scen, y_true, weather_future

    def run(self, out_dir, deploy_start, deploy_end, M, warmup_days=0):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        start = pd.Timestamp(deploy_start) - pd.Timedelta(days=warmup_days)
        days = pd.date_range(start, deploy_end, freq="D")

        durations = []
        for day in tqdm(days, desc=f"deploy -> {out_dir.name}"):
            path = out_dir / f"{day.date()}.npz"
            if path.exists():  # resume support
                continue
            t0 = time.time()
            scen, y_true, weather_future = self.sample_day(day, M)
            np.savez_compressed(path, scenarios=scen, y_true=y_true,
                                weather_future=weather_future, date=str(day.date()))
            durations.append(time.time() - t0)

        manifest = {
            "n_days": len(days),
            "deploy_start": str(deploy_start), "deploy_end": str(deploy_end),
            "warmup_days": warmup_days, "M": M,
            "n_sampled_this_run": len(durations),
            "total_sampling_seconds": float(np.sum(durations)),
            "mean_seconds_per_day": float(np.mean(durations)) if durations else None,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Cached {len(days)} days in {out_dir} "
              f"({len(durations)} newly sampled)")
        return manifest
