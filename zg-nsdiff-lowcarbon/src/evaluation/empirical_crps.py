"""Vectorised empirical-sample CRPS (spec 2.4).

CRPS = 1/S sum_s |x_s - y| - 1/(2 S^2) sum_{s,r} |x_s - x_r|

The pairwise term uses the sorted identity
sum_{s,r} |x_s - x_r| = 2 * sum_i (2i - S + 1) x_(i)   (0-indexed order stats)
giving O(S log S) instead of O(S^2).
"""
import numpy as np
import torch


def crps_samples_torch(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """samples [..., S], truth [...] -> CRPS [...] (elementwise)."""
    S = samples.shape[-1]
    term1 = (samples - truth.unsqueeze(-1)).abs().mean(dim=-1)
    sorted_s, _ = torch.sort(samples, dim=-1)
    idx = torch.arange(S, device=samples.device, dtype=samples.dtype)
    weights = 2.0 * idx - S + 1.0
    pairwise_sum = 2.0 * (sorted_s * weights).sum(dim=-1)  # == sum_{s,r} |x_s - x_r|
    term2 = pairwise_sum / (2.0 * S * S)
    return term1 - term2


def crps_samples(samples, truth):
    if isinstance(samples, np.ndarray):
        return (
            crps_samples_torch(torch.from_numpy(np.ascontiguousarray(samples)).double(),
                               torch.from_numpy(np.ascontiguousarray(truth)).double())
            .numpy()
        )
    return crps_samples_torch(samples, truth)
