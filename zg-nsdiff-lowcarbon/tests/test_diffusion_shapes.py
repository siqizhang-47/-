import torch

from src.models.diffusion_utils import (
    DiffusionSchedule,
    cal_forward_noise,
    cal_sigma_tilde,
    q_sample,
)
from src.models.NsDiffExo import NsDiffExo


def make_sched():
    return DiffusionSchedule(6, "linear", 1e-4, 1e-2, torch.device("cpu"))


def test_q_sample_shape_preserved():
    sched = make_sched()
    y = torch.randn(3, 8, 3)
    t = torch.randint(0, 6, (3,))
    noise = torch.randn_like(y)
    out = q_sample(y, y.clone(), sched, t, noise)
    assert out.shape == y.shape


def test_forward_noise_nonnegative():
    sched = make_sched()
    gx = torch.rand(3, 8, 3) + 0.01
    y_sigma = torch.rand(3, 8, 3) + 0.01
    for ti in range(6):
        t = torch.full((3,), ti, dtype=torch.long)
        assert (cal_forward_noise(sched, gx, y_sigma, t) >= 0).all()
        assert (cal_sigma_tilde(sched, gx, y_sigma, t) > 0).all()


def test_sampling_finite_and_chunk_equivalence(tiny_cfg, tiny_batch):
    torch.manual_seed(0)
    model = NsDiffExo(tiny_cfg, torch.device("cpu"))
    model.eval()
    g = torch.Generator().manual_seed(7)
    s1 = model.sample_trajectories(tiny_batch, num_samples=8, chunk_size=4, generator=g)
    assert s1.shape == (3, 8, 3, 8)
    assert torch.isfinite(s1).all()
    g2 = torch.Generator().manual_seed(7)
    s2 = model.sample_trajectories(tiny_batch, num_samples=8, chunk_size=8, generator=g2)
    assert torch.isfinite(s2).all()
    assert (s1.mean() - s2.mean()).abs() < 1.0


def test_element_losses_shapes(tiny_cfg, tiny_batch):
    torch.manual_seed(0)
    model = NsDiffExo(tiny_cfg, torch.device("cpu"))
    elems = model.element_losses(tiny_batch)
    B, H = 3, 8
    for key in ["mean_elem", "variance_elem", "diffusion_elem"]:
        assert elems[key].shape == (B, H, 3)
        assert torch.isfinite(elems[key]).all()
    assert (elems["reverse_variance_elem"] >= -1e-6).all()


def test_deepvar_loss_and_shapes(tiny_cfg, tiny_batch):
    from src.baselines.deepvar_adapter import DeepVAR
    torch.manual_seed(0)
    model = DeepVAR(tiny_cfg, torch.device("cpu"))
    loss = model.loss(tiny_batch)
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    s = model.sample(tiny_batch, num_samples=5, chunk_size=2)
    assert s.shape == (3, 8, 3, 5)
    assert torch.isfinite(s).all()
