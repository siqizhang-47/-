"""Baseline 5: original COSA, ported to the scenario-ensemble setting.

Strictly follows the source engineering (COSA_ICLR2026-master/tta/cosa.py
SimpleOutputAdapter): context = recent buffer target means
(BUFFER_CONTEXT_SIZE=5), per-variable tanh(g)-gated linear correction with
zero-initialized gate, MSE objective. The correction is applied to the
scenario MEAN and shared as a translation across all scenarios (COSA is a
point-forecast corrector; the shared shift is the faithful lift to a
generative model -- documented deviation, see README).

It has NO scale channel: pure translation.
"""
import torch
import torch.nn as nn


class COSAOriginalAdapter(nn.Module):
    def __init__(self, H=24, C=4, buffer_context_size=5):
        super().__init__()
        self.H, self.C = H, C
        self.buffer_context_size = buffer_context_size
        # VAR_WISE_GATING=True: one linear per variable, input = pred + context
        self.fc_layers = nn.ModuleList([
            nn.Linear(H + buffer_context_size, H) for _ in range(C)])
        self.gate = nn.Parameter(torch.zeros(C))
        for fc in self.fc_layers:
            nn.init.xavier_uniform_(fc.weight, gain=0.1)
            nn.init.zeros_(fc.bias)

    def compute_shift(self, mu, ctx):
        """mu (H, C) scenario mean, ctx (buffer_context_size,) -> shift (H, C)."""
        corrections = []
        for c in range(self.C):
            inp = torch.cat([mu[:, c], ctx])
            corrections.append(self.fc_layers[c](inp))
        correction = torch.stack(corrections, dim=-1)        # (H, C)
        return torch.tanh(self.gate)[None, :] * correction

    def forward(self, scenarios, ctx):
        """scenarios (M, H, C), ctx (buffer_context_size,) recent target means."""
        mu = scenarios.mean(0)
        shift = self.compute_shift(mu, ctx)
        return scenarios + shift[None]


def cosa_loss(adapter, scenarios, y_true, ctx, l2_weight=1e-4):
    """Original COSA objective: MSE (on the corrected point forecast, i.e. the
    scenario mean) + L2 on adapter parameters, exactly as in the source."""
    adapted = adapter(scenarios, ctx)
    loss = torch.nn.functional.mse_loss(adapted.mean(0), y_true)
    l2 = sum(p.pow(2).sum() for p in adapter.parameters() if p.requires_grad)
    return loss + l2_weight * l2
