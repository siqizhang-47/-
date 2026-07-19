"""Trailing window variance, kept identical to the original NsDiff implementation."""
import torch
import torch.nn.functional as F


def wv_sigma_trailing(x_enc: torch.Tensor, window_size: int, discard_rep: bool = False) -> torch.Tensor:
    """Variance over the trailing window [t - window_size, t - 1] for each step.

    x_enc: (B, T, N) -> sigma: (B, T', N); with discard_rep=True the leading
    replicate-padded region is dropped (T' = T - window_size + 1).
    """
    if x_enc.dim() != 3:
        raise ValueError("x_enc must be (B, T, N)")
    B, T, N = x_enc.shape
    if window_size < 1 or window_size > T:
        raise ValueError(f"window_size must be in [1, {T}], got {window_size}")
    if not discard_rep:
        x_enc = F.pad(x_enc, (0, 0, window_size, 0), mode="replicate")
    windows = x_enc.unfold(1, window_size, 1)
    return windows.var(dim=3, unbiased=False)
