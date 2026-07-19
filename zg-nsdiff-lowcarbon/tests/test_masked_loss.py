import torch

from src.layer.masked_reduction import masked_mean
from src.models.ZGNsDiff import build_magnitude_mask


def test_masked_mean_ignores_inactive():
    loss = torch.tensor([[1.0, 100.0], [2.0, 200.0]])
    mask = torch.tensor([[True, False], [True, False]])
    assert masked_mean(loss, mask).item() == 1.5


def test_masked_mean_no_nan_when_empty():
    loss = torch.rand(4, 5)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    out = masked_mean(loss, mask)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_magnitude_mask_rules():
    B, H = 2, 3
    observed = torch.ones(B, H, 4, dtype=torch.bool)
    active = torch.zeros(B, H, 3, dtype=torch.bool)
    active[:, :, 1] = True  # only heating active
    mask = build_magnitude_mask(observed, active)
    assert mask[..., 0].all()            # electricity always participates
    assert not mask[..., 1].any()        # cooling inactive -> excluded
    assert mask[..., 2].all()            # heating active -> included
    assert not mask[..., 3].any()        # pv inactive -> excluded


def test_unobserved_never_participates():
    B, H = 2, 3
    observed = torch.ones(B, H, 4, dtype=torch.bool)
    observed[0, :, 2] = False            # heating unobserved for sample 0
    active = torch.ones(B, H, 3, dtype=torch.bool)
    mask = build_magnitude_mask(observed, active)
    assert not mask[0, :, 2].any()
    assert mask[1, :, 2].all()


def test_changing_inactive_prediction_does_not_change_loss():
    B, H = 2, 4
    observed = torch.ones(B, H, 4, dtype=torch.bool)
    active = torch.zeros(B, H, 3, dtype=torch.bool)
    mask = build_magnitude_mask(observed, active)
    target = torch.randn(B, H, 4)
    pred = torch.randn(B, H, 4)
    l1 = masked_mean((pred - target).square(), mask)
    pred2 = pred.clone()
    pred2[..., 1:] += 100.0              # perturb only zero-inflated coords
    l2 = masked_mean((pred2 - target).square(), mask)
    assert torch.allclose(l1, l2)
    pred3 = pred.clone()
    pred3[..., 0] += 1.0                 # perturbing electricity must change it
    l3 = masked_mean((pred3 - target).square(), mask)
    assert not torch.allclose(l1, l3)
