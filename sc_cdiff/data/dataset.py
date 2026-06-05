"""Step 5: Datasets. Training uses sliding windows (phase augmentation); val/test
use aligned natural days. All conditioning fields are sliced from the continuous
tape produced in step 1.

A training window is SKIPPED if it overlaps any FILLED (earthquake-imputed) hour,
so the model never fits fabricated values. Evaluation drops filled days too.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..normalize import Normalizer


def _normalize_weather(W, w_mu, w_sd):
    # W: [6,24]
    return (W - w_mu[:, None]) / w_sd[:, None]


class _Base(Dataset):
    def __init__(self, cfg):
        paths = cfg["paths"]
        npz = np.load(os.path.join(paths["artifacts"], paths["dataset_npz"]))
        self.Yfull = npz["Yfull"]            # [n,5]
        self.Wfull = npz["Wfull"]            # [n,6]
        self.CALfull = npz["CALfull"]        # [n,8]
        self.ERAfull = npz["ERAfull"]        # [n]
        self.IRRfull = npz["IRRfull"]        # [n]
        self.FILLEDfull = npz["FILLEDfull"]  # [n]
        self.train_end = int(npz["train_end_hour"])
        self.val_end = int(npz["val_end_hour"])
        self.n = int(npz["n_hours"])
        self.Yhat = np.load(os.path.join(paths["artifacts"], paths["yhat_npy"]))   # [n,5]
        wpath = paths["woracle_npy"] if cfg.get("use_oracle_weather", False) else paths["what_npy"]
        self.What = np.load(os.path.join(paths["artifacts"], wpath))               # [n,6]
        self.norm = Normalizer.load(os.path.join(paths["artifacts"], paths["norm_stats"]))
        self.w_mu = np.asarray(self.norm.stats["W"][0], dtype=np.float32)
        self.w_sd = np.asarray(self.norm.stats["W"][1], dtype=np.float32)
        self.H = cfg["horizon"]
        self.c_idx, self.h_idx, self.pv_idx = cfg["c_idx"], cfg["h_idx"], cfg["pv_idx"]

    def _make_item(self, s, k):
        H = self.H
        sl = slice(s, s + H)
        era = int(self.ERAfull[s])
        Yraw = self.Yfull[sl].T                                  # [5,24]
        Y = self.norm.normalize_np(Yraw, era)                    # standardized target
        W = _normalize_weather(self.What[sl].T.astype(np.float32), self.w_mu, self.w_sd)
        CAL = self.CALfull[sl].T.astype(np.float32)              # [8,24]
        Yhat = self.norm.normalize_np(self.Yhat[sl].T, era)      # condition, same space
        irr = self.IRRfull[sl].astype(np.float32)                # [24] true irradiance

        # observation mask M: first k hours observed (all channels)
        M = np.zeros((5, H), dtype=np.float32)
        if k > 0:
            M[:, :k] = 1.0
        # validity mask m: exclude structural-zero cells from the diffusion loss
        m = np.ones((5, H), dtype=np.float32)
        m[self.pv_idx] = (irr > 0).astype(np.float32)
        m[self.c_idx] = (Yraw[self.c_idx] > 0).astype(np.float32)
        m[self.h_idx] = (Yraw[self.h_idx] > 0).astype(np.float32)
        # gate labels (cool/heat activation), only meaningful in generated region
        Gc = (Yraw[self.c_idx] > 0).astype(np.float32)
        Gh = (Yraw[self.h_idx] > 0).astype(np.float32)

        return {
            "Y": torch.from_numpy(Y),
            "Yraw": torch.from_numpy(Yraw.astype(np.float32)),
            "W": torch.from_numpy(W),
            "CAL": torch.from_numpy(CAL),
            "Yhat": torch.from_numpy(Yhat),
            "era": torch.tensor(era, dtype=torch.long),
            "irr": torch.from_numpy(irr),
            "M": torch.from_numpy(M),
            "m": torch.from_numpy(m),
            "Gc": torch.from_numpy(Gc),
            "Gh": torch.from_numpy(Gh),
            "k": torch.tensor(k, dtype=torch.long),
            "start": torch.tensor(s, dtype=torch.long),
        }


class TrainWindows(_Base):
    """Sliding 24h frame over the training tape, stride from config. Skips any
    window overlapping a filled hour. k (forecast horizon) is randomized."""

    def __init__(self, cfg):
        super().__init__(cfg)
        stride = cfg["window"]["stride"]
        H = self.H
        starts = []
        for s in range(0, self.train_end - H + 1, stride):
            if self.FILLEDfull[s:s + H].any():
                continue
            starts.append(s)
        self.starts = starts
        self.ks = [0, 6, 12, 18]
        print(f"[TrainWindows] stride={stride} windows={len(self.starts)} (filled skipped)")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        k = int(np.random.choice(self.ks))
        return self._make_item(s, k)


class AlignedDays(_Base):
    """Natural days (00:00 start, non-overlapping) for a split, fixed k for eval.
    split in {'val','test'}. Filled days excluded by default."""

    def __init__(self, cfg, split, k, exclude_filled=True):
        super().__init__(cfg)
        H = self.H
        if split == "val":
            lo, hi = self.train_end, self.val_end
        elif split == "test":
            lo, hi = self.val_end, self.n
        elif split == "train":
            lo, hi = 0, self.train_end
        else:
            raise ValueError(split)
        # align lo to a day boundary (the tape starts at 00:00 so multiples of 24)
        lo = ((lo + H - 1) // H) * H
        starts = []
        for s in range(lo, hi - H + 1, H):
            if exclude_filled and self.FILLEDfull[s:s + H].any():
                continue
            starts.append(s)
        self.starts = starts
        self.k = k
        print(f"[AlignedDays:{split}] k={k} days={len(self.starts)}")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        return self._make_item(self.starts[i], self.k)
