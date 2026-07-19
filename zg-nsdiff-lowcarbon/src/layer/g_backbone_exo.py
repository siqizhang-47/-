"""Variance network g_psi with future exogenous increments (spec 7.4).

Keeps the original trailing-variance + MLP extrapolation path, adds an
additive delta computed from the oracle future conditions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.sigma import wv_sigma_trailing

EPS = 1e-8


class SigmaEstimationExo(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, condition_dim,
                 hidden_size=512, kernel_size=24, cond_hidden=64):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.kernel_size = kernel_size
        self.history_mlp = nn.Sequential(
            nn.Linear(seq_len - kernel_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, enc_in),
        )

    def forward(self, history_target, history_condition, future_condition):
        # history_condition is accepted for interface parity (optional pooling);
        # the mandatory exogenous path is future_condition (spec 7.4).
        B, T, N = history_target.shape
        sigma = wv_sigma_trailing(history_target, self.kernel_size, discard_rep=True)
        sigma = sigma[:, -(T - self.kernel_size):, :] + EPS
        base_logits = self.history_mlp(sigma.permute(0, 2, 1))       # [B, N, H]
        base_logits = base_logits.permute(0, 2, 1)[:, -self.pred_len:, :]  # [B, H, N]
        future_delta = self.condition_mlp(future_condition)          # [B, H, N]
        pred_sigma = F.softplus(base_logits + future_delta) + EPS
        return pred_sigma
