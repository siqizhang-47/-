"""Per-channel normalization shared by dataset / training / sampling.

Conventions (plan §5.1, step 2):
  PV  : standardized with per-era daytime stats (era passed in)
  E   : standardized with all-hour stats
  C/H : standardized with activated-only (val>0) stats -- zeros map to a
        negative value but those positions are excluded from the diffusion
        loss via the validity mask `m`, and re-zeroed at sampling by the gate.
  HW  : standardized with all-hour stats
"""
from __future__ import annotations

import json

import numpy as np


class Normalizer:
    def __init__(self, stats: dict):
        self.stats = stats
        self.per_era_pv = {int(k): v for k, v in stats["per_era_pv"].items()}
        self.E = stats["E"]
        self.C = stats["C"]
        self.H = stats["H"]
        self.HW = stats["HW"]

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path) as f:
            return cls(json.load(f))

    def _pv(self, era: int):
        return self.per_era_pv.get(int(era), self.per_era_pv[2])

    def normalize_np(self, Y: np.ndarray, era: int) -> np.ndarray:
        """Y: [5,24] raw -> standardized. era is a scalar int for this window."""
        out = np.empty_like(Y, dtype=np.float32)
        mu, sd = self._pv(era)
        out[0] = (Y[0] - mu) / sd
        out[1] = (Y[1] - self.E[0]) / self.E[1]
        out[2] = (Y[2] - self.C[0]) / self.C[1]
        out[3] = (Y[3] - self.H[0]) / self.H[1]
        out[4] = (Y[4] - self.HW[0]) / self.HW[1]
        return out

    def denormalize_torch(self, Y, era):
        """Y: [B,5,24] torch tensor, era: [B] long tensor. Returns raw scale."""
        import torch
        dev = Y.device
        pv_mu = torch.tensor([self._pv(int(e))[0] for e in era.tolist()], device=dev)
        pv_sd = torch.tensor([self._pv(int(e))[1] for e in era.tolist()], device=dev)
        out = torch.empty_like(Y)
        out[:, 0] = Y[:, 0] * pv_sd[:, None] + pv_mu[:, None]
        out[:, 1] = Y[:, 1] * self.E[1] + self.E[0]
        out[:, 2] = Y[:, 2] * self.C[1] + self.C[0]
        out[:, 3] = Y[:, 3] * self.H[1] + self.H[0]
        out[:, 4] = Y[:, 4] * self.HW[1] + self.HW[0]
        return out
