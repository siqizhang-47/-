"""Phase 1 acceptance: forward shapes + weather-conditioning smoke test."""
from pathlib import Path

import pytest
import torch

from utils.config import Config

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(**over):
    cfg = Config(
        data_name="energy", task="MW", cont_len=48, pred_len=24,
        num_feat=8, num_target=4, freq="h",
        n_emb=2, cont_hidden_dim=32, pred_hidden_dim=32, step_hidden_dim=32,
        time_hidden_dim=32, n_depth=2, n_heads=4, attn_dropout=0.1,
        mlp_ratio=1, non_attn=False,
        weather_cond_mode="token", use_time_cond=True, use_window_norm=True,
        n_steps=10, beta_start=0.0001, beta_end=0.5, beta_schedule="quad",
        parameterization="noise", step_dist="uniform",
        use_contrast="non-contrast", contrast_weight=0.0, n_negatives=4,
        temperature=0.1, train_batch_size=4, n_epochs=1, init_lr=1e-3,
        device="cpu",
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_batch(B=3, cfg=None):
    cfg = cfg or make_cfg()
    x = torch.randn(B, cfg.cont_len, cfg.num_feat)
    y = torch.randn(B, cfg.pred_len, cfg.num_target)
    w = torch.randn(B, cfg.pred_len, cfg.num_feat - cfg.num_target)
    x_mark = torch.randn(B, cfg.cont_len, 4)
    y_mark = torch.randn(B, cfg.pred_len, 4)
    k = torch.randint(0, cfg.n_steps, (B,))
    return x, y, w, x_mark, y_mark, k


@pytest.mark.parametrize("mode", ["token", "pooled"])
def test_denoiser_output_shape(mode):
    from backbone.network import Denoiser

    cfg = make_cfg(weather_cond_mode=mode)
    net = Denoiser(cfg)
    x, y, w, x_mark, y_mark, k = make_batch(3, cfg)
    out = net(x, y, k, w, x_mark, y_mark)
    assert out.shape == (3, 24, 4)


def test_sampling_shape():
    from backbone.model import DiffMTS

    cfg = make_cfg()
    model = DiffMTS(cfg)
    x, y, w, x_mark, y_mark, _ = make_batch(2, cfg)
    scen = model.pred_sampling(x, w, n_samples=5, x_mark=x_mark, y_mark=y_mark)
    assert scen.shape == (10, 24, 4)


def test_window_norm_targets_only():
    from backbone.model import DiffMTS

    cfg = make_cfg()
    model = DiffMTS(cfg)
    x, y, _, _, _, _ = make_batch(4, cfg)
    x_n, y_n, mean, std = model.instance_normalization(x, y)
    # target channels normalized to ~N(0,1) over the window
    assert x_n[:, :, :4].mean(dim=1).abs().max() < 1e-4
    # covariate channels untouched
    assert torch.allclose(x_n[:, :, 4:], x[:, :, 4:])
    # denormalization inverts
    scen = y_n.repeat_interleave(1, dim=0)
    back = model.instance_denormalization(scen, mean, std)
    assert torch.allclose(back, y, atol=1e-5)


def test_train_step_and_early_stopper():
    from utils.early_stopping import EarlyStopper

    stopper = EarlyStopper(patience=2)
    assert not stopper.update(1.0, 0)
    assert not stopper.update(0.9, 1)
    assert not stopper.update(0.95, 2)
    assert stopper.update(0.96, 3)          # 2 bad evaluations -> stop
    assert stopper.best_step == 1


@pytest.mark.slow
def test_ghi_conditioning_direction():
    """Zeroing future GHI must significantly lower daytime PV scenario means.
    Needs a trained checkpoint; run after Phase 1."""
    ckpt = ROOT / "checkpoints/backbone_wnorm_on.pt"
    if not ckpt.exists():
        pytest.skip("train the backbone first")
    import numpy as np

    from backbone.data_loader import Dataset_MTS
    from backbone.model import DiffMTS
    from utils.config import load_config

    cfg = load_config(ROOT / "configs/base.yaml", ROOT / "configs/backbone_energy.yaml")
    cfg.device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model = DiffMTS(cfg)
    model.load_weights(str(ckpt))

    ds = Dataset_MTS("energy", str(ROOT / "data/processed/energy.csv"),
                     cfg.cont_len, cfg.pred_len, status="val")
    # a summer noon-anchored window (index chosen so pred covers daytime)
    idx = 180 * 24 + 8 - cfg.cont_len  # forecast start 08:00, day 180 of 2019
    batch = [torch.from_numpy(np.asarray(b)).unsqueeze(0).float().to(cfg.device)
             for b in ds[idx]]
    x, y0, w, x_mark, y_mark = batch
    x_n, _, mean, std = model.instance_normalization(x, y0)
    scen = model.pred_sampling(x_n, w, 20, x_mark, y_mark)
    scen = model.instance_denormalization(scen, mean, std)
    w_zero = w.clone()
    ghi_col = 3
    w_zero[:, :, ghi_col] = (0.0 - ds.scaler_cov.mean_[ghi_col]) / ds.scaler_cov.scale_[ghi_col]
    scen0 = model.pred_sampling(x_n, w_zero, 20, x_mark, y_mark)
    scen0 = model.instance_denormalization(scen0, mean, std)
    pv, pv0 = scen[:, :, 0].mean().item(), scen0[:, :, 0].mean().item()
    assert pv0 < pv, f"PV mean should drop when GHI is zeroed ({pv0=} vs {pv=})"
