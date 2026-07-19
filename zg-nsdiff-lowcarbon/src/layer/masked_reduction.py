"""Masked reductions for the ZG losses (spec 8.5)."""
import torch


def masked_mean(loss_elem: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weight = mask.to(loss_elem.dtype)
    return (loss_elem * weight).sum() / weight.sum().clamp_min(1.0)


def variable_balanced_masked_mean(loss_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-variable masked mean, then average over variables (ablation only)."""
    weight = mask.to(loss_elem.dtype)
    per_var_num = (loss_elem * weight).sum(dim=tuple(range(loss_elem.dim() - 1)))
    per_var_den = weight.sum(dim=tuple(range(loss_elem.dim() - 1))).clamp_min(1.0)
    return (per_var_num / per_var_den).mean()
