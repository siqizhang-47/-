"""Copied verbatim from NsDiff-main/src/utils/sigma.py (dead commented-out code
at the bottom of the original file removed).

``wv_sigma_trailing`` produces the empirical target variance sequence that
NsDiff's forward diffusion process needs (``y_sigma`` in the training loop).
"""
import torch
import torch.nn.functional as F


def wv_sigma(x_enc, window_size):
    """
    Compute the variance over a sliding window along the T dimension.

    For each time step t, the variance is calculated over a window of size `window_size`
    centered around t. For even window sizes, the window is asymmetrically padded to maintain
    the same output length as the input.

    Args:
        x_enc (Tensor): Input tensor of shape (B, T, N)
        window_size (int): Size of the sliding window

    Returns:
        sigma (Tensor): Variance tensor of shape (B, T, N)
    """
    B, T, N = x_enc.shape
    if window_size % 2 == 0:
        pad_left = window_size // 2
        pad_right = window_size // 2 - 1
    else:
        pad_left = pad_right = window_size // 2
    x_padded = F.pad(x_enc, (0, 0, pad_left, pad_right), mode='replicate')
    windows = x_padded.unfold(dimension=1, size=window_size, step=1)

    sigma = windows.var(dim=3, unbiased=False)  # Shape: (B, T, N)

    return sigma


def wv_sigma_trailing(x_enc, window_size, discard_rep=False):
    """
    Compute the variance over a trailing window for each time step.

    For each time step t, the variance is calculated over the window [t - window_size, t - 1].

    Args:
        x_enc (Tensor): Input tensor of shape (B, T, N)
        window_size (int): Size of the trailing window

    Returns:
        sigma (Tensor): Variance tensor of shape (B, T, N)
    """
    if not isinstance(x_enc, torch.Tensor):
        raise TypeError("x_enc must be a torch.Tensor")

    if x_enc.dim() != 3:
        raise ValueError("x_enc must be a 3D tensor with shape (B, T, N)")

    B, T, N = x_enc.shape

    if window_size < 1 or window_size > T:
        raise ValueError(f"window_size must be between 1 and T (got window_size={window_size}, T={T})")

    if not discard_rep:
        x_enc = F.pad(x_enc, (0, 0, window_size, 0), mode='replicate')  # Shape: (B, T + window_size, N)

    windows = x_enc.unfold(1, window_size, 1)

    sigma = windows.var(dim=3, unbiased=False)  # Shape: (B, T, N)
    return sigma
