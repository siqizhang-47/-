"""Residual context features for the location-scale adapter (v4 §7).

Maintains a buffer of the most recent K days of residuals
    e = y_true - scenario_mean            (H, C)
plus the daily predictive std (H, C), and produces a flat context feature of
dimension D_ctx = 12 + 96 + 12 = 120 (for H=24, C=4):

  location:   per-carrier residual mean / median / trend slope     (3*C = 12)
              horizon-wise bias (mean e over buffer, flattened)    (H*C = 96)
  dispersion: per-carrier residual std                             (C)
              empirical/predictive dispersion ratio r_c            (C)
              recent MAE                                           (C)
"""
from collections import deque

import torch
import torch.nn.functional as F


class ResidualContext:
    def __init__(self, K: int, H: int = 24, C: int = 4):
        self.K = K
        self.H = H
        self.C = C
        self.buffer = deque(maxlen=K)

    def __len__(self):
        return len(self.buffer)

    @property
    def d_ctx(self):
        return 3 * self.C + self.H * self.C + 3 * self.C

    def push(self, y_true, scenario_mean, scenario_std):
        """Store residual e = y_true - mu and the day's predictive std (H, C)."""
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        mu = torch.as_tensor(scenario_mean, dtype=torch.float32)
        std = torch.as_tensor(scenario_std, dtype=torch.float32)
        self.buffer.append({"e": y_true - mu, "pred_std": std, "y_true": y_true})

    def recent_target_means(self, n: int) -> torch.Tensor:
        """Last n daily target means (scalar per day), padded with the oldest
        available value -- used by the original-COSA baseline context."""
        means = [d["y_true"].mean().item() for d in list(self.buffer)[-n:]][::-1]
        if not means:
            means = [0.0]
        while len(means) < n:
            means.append(means[-1])
        return torch.tensor(means, dtype=torch.float32)

    def features(self) -> torch.Tensor:
        assert len(self.buffer) > 0, "context buffer is empty (cold start not done?)"
        e = torch.stack([d["e"] for d in self.buffer])            # (K', H, C)
        pred_std = torch.stack([d["pred_std"] for d in self.buffer])

        # location statistics
        mean_c = e.mean(dim=(0, 1))                               # (C,)
        median_c = e.reshape(-1, self.C).median(dim=0).values     # (C,)
        daily_mean = e.mean(dim=1)                                # (K', C)
        k = daily_mean.shape[0]
        if k > 1:
            t = torch.arange(k, dtype=torch.float32)
            t = t - t.mean()
            slope_c = (t[:, None] * (daily_mean - daily_mean.mean(0))).sum(0) / (t ** 2).sum()
        else:
            slope_c = torch.zeros(self.C)
        horizon_bias = e.mean(dim=0).reshape(-1)                  # (H*C,)

        # dispersion statistics
        std_c = e.reshape(-1, self.C).std(dim=0, unbiased=False)  # (C,)
        ratio_c = std_c / (pred_std.mean(dim=(0, 1)) + 1e-8)      # (C,)
        mae_c = e.abs().mean(dim=(0, 1))                          # (C,)

        # grouped normalization: a single layer_norm over all 120 dims would let
        # the 96 horizon-bias features dilute the 12 dispersion features and
        # destroy their absolute level (the direct supervision signal for s).
        # - horizon bias: layer_norm within the group (its SHAPE is the signal;
        #   the absolute level is already carried by mean_c/median_c)
        # - scalar stats: kept in raw train-scaler units (already O(1));
        #   dispersion ratio passed as log(r_c) so 0 == calibrated
        horizon_bias = F.layer_norm(horizon_bias, horizon_bias.shape)
        log_ratio_c = torch.log(ratio_c + 1e-8)
        feat = torch.cat([mean_c, median_c, slope_c, horizon_bias,
                          std_c, log_ratio_c, mae_c])
        assert feat.shape[0] == self.d_ctx
        return torch.clamp(feat, -10.0, 10.0)
