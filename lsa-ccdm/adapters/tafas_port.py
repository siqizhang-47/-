"""Baseline 4: TAFAS, ported to daily scenario-ensemble granularity.

Reference: COSA_ICLR2026-master/tta/tafas.py (TAFAS, AAAI 2025). TAFAS's core
mechanism is a gated calibration module (GCM) that corrects the model output
using arrived ground truth. Daily-granularity port: a per-variable linear
calibration of the scenario-mean trajectory, gated (zero-initialized) and
trained with MSE on the arrived-truth buffer. The correction is applied as a
shared translation to all scenarios -- a reasonable lift of a point-forecast
TTA method to a generative model (documented in the paper).
"""
import torch
import torch.nn as nn


class TAFASAdapter(nn.Module):
    def __init__(self, H=24, C=4):
        super().__init__()
        self.H, self.C = H, C
        # gated calibration module: per-variable linear on the predicted trajectory
        self.fc_layers = nn.ModuleList([nn.Linear(H, H) for _ in range(C)])
        self.gate = nn.Parameter(torch.zeros(C))
        for fc in self.fc_layers:
            nn.init.xavier_uniform_(fc.weight, gain=0.1)
            nn.init.zeros_(fc.bias)

    def compute_shift(self, mu):
        corrections = [self.fc_layers[c](mu[:, c]) for c in range(self.C)]
        correction = torch.stack(corrections, dim=-1)  # (H, C)
        return torch.tanh(self.gate)[None, :] * correction

    def forward(self, scenarios, ctx=None):
        """scenarios (M, H, C); ctx unused (TAFAS conditions on the prediction only)."""
        mu = scenarios.mean(0)
        shift = self.compute_shift(mu)
        return scenarios + shift[None]


def tafas_loss(adapter, scenarios, y_true, ctx=None):
    adapted = adapter(scenarios, ctx)
    return torch.nn.functional.mse_loss(adapted.mean(0), y_true)
