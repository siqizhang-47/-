import torch

from src.data.target_transform import TargetTransform
from src.models.ZGNsDiff import ZGNsDiff, apply_gate_and_inverse


def test_gate_shapes_and_range(tiny_cfg, tiny_batch):
    torch.manual_seed(0)
    model = ZGNsDiff(tiny_cfg, torch.device("cpu"))
    model.eval()
    prob = model.predict_gate_prob(tiny_batch)
    B, H = tiny_batch["future_target"].shape[:2]
    assert prob.shape == (B, H, 3)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_gate_sampling_reproducible():
    prob = torch.full((2, 4, 3), 0.5)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s1 = ZGNsDiff.sample_gates(prob, 10, g1)
    s2 = ZGNsDiff.sample_gates(prob, 10, g2)
    assert torch.equal(s1, s2)
    assert s1.shape == (2, 4, 3, 10)


def test_gate_zero_gives_exact_zero(tiny_stats):
    tf = TargetTransform(tiny_stats)
    B, H, S = 2, 4, 6
    magnitude = torch.rand(B, H, 4, S) * 2
    gates = torch.zeros(B, H, 3, S)
    raw = apply_gate_and_inverse(magnitude, gates, tf)
    assert (raw[..., 1:, :] == 0.0).all()          # exact zeros
    assert (raw[..., 0, :] >= 0).all()             # electricity clamped >= 0


def test_gate_one_gives_nonnegative_magnitude(tiny_stats):
    tf = TargetTransform(tiny_stats)
    B, H, S = 2, 4, 6
    magnitude = torch.randn(B, H, 4, S)            # includes negatives
    gates = torch.ones(B, H, 3, S)
    raw = apply_gate_and_inverse(magnitude, gates, tf)
    assert (raw >= 0).all()
