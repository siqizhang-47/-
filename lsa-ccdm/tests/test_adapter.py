"""Phase 3 acceptance: identity start, correlation preservation, scale
identifiability (CRPS vs MSE), no-look-ahead ordering."""
import numpy as np
import pytest
import torch
from scipy.stats import pearsonr, spearmanr

from adapters.context import ResidualContext
from adapters.lsa import LSAdapter, ensemble_crps, lsa_loss


def make_ctx_features(K=7):
    ctx = ResidualContext(K=K)
    for _ in range(K):
        ctx.push(torch.randn(24, 4), torch.randn(24, 4), torch.rand(24, 4) + 0.5)
    return ctx.features()


# 1. identity at initialization ------------------------------------------------
def test_identity_start():
    adapter = LSAdapter()
    scen = torch.randn(50, 24, 4) * 3 + 1
    ctx = make_ctx_features()
    out = adapter(scen, ctx)
    assert (out - scen).abs().max() < 1e-5


def test_frozen_variants_have_no_grad():
    a6 = LSAdapter(s_frozen=True)     # baseline 6 shift-only
    assert not a6.s_head.weight.requires_grad
    a7 = LSAdapter(delta_frozen=True)  # baseline 7 scale-only
    assert not a7.delta_head.weight.requires_grad and not a7.g.requires_grad
    scen = torch.randn(20, 24, 4)
    ctx = make_ctx_features()
    _, delta, s = a7.forward_with_params(scen, ctx)
    assert delta.abs().max() == 0
    _, delta, s = a6.forward_with_params(scen, ctx)
    assert torch.allclose(s, torch.ones(4))


def test_pred_context_changes_delta():
    """the adapter must SEE today's predicted trajectory: different scenario
    means under the same residual context -> different Delta."""
    adapter = LSAdapter()
    with torch.no_grad():  # non-trivial weights so outputs depend on inputs
        adapter.g.copy_(torch.ones(4))
        adapter.delta_head.weight.normal_(0, 0.5)
    ctx = make_ctx_features()
    scen_a = torch.randn(30, 24, 4)
    scen_b = scen_a + torch.sin(torch.arange(24.0))[None, :, None]  # other shape
    d_a, _ = adapter.compute_params(ctx, scen_a)
    d_b, _ = adapter.compute_params(ctx, scen_b)
    assert (d_a - d_b).abs().max() > 1e-4


# 2. correlation preservation --------------------------------------------------
def test_correlation_preservation():
    adapter = LSAdapter()
    # random non-trivial (Delta, s)
    with torch.no_grad():
        adapter.g.copy_(torch.randn(4))
        adapter.s_head.weight.normal_(0, 0.5)
        adapter.delta_head.weight.normal_(0, 0.5)
        adapter.delta_head.bias.normal_(0, 1.0)
    # correlated scenarios
    L = torch.tensor([[1.0, 0, 0, 0], [0.8, 0.6, 0, 0],
                      [0.3, -0.4, 0.86, 0], [-0.2, 0.5, 0.1, 0.83]])
    scen = torch.randn(500, 24, 4) @ L.T
    ctx = make_ctx_features()
    out = adapter(scen, ctx).detach().numpy()
    scen = scen.numpy()
    for h in range(0, 24, 6):
        for i in range(4):
            for j in range(i + 1, 4):
                rp0 = pearsonr(scen[:, h, i], scen[:, h, j])[0]
                rp1 = pearsonr(out[:, h, i], out[:, h, j])[0]
                rs0 = spearmanr(scen[:, h, i], scen[:, h, j])[0]
                rs1 = spearmanr(out[:, h, i], out[:, h, j])[0]
                assert abs(rp0 - rp1) < 1e-4
                assert abs(rs0 - rs1) < 1e-4


# 3. scale identifiability: CRPS yes, MSE no (v4 §9 core claim) ---------------
# IMPORTANT: this test runs with the PRODUCTION lambda defaults imported from
# run_adapter.py -- if a default change re-pins the scale channel, this test
# must fail (guards against test/production divergence).
def _train_scale(objective, n_steps=200):
    from experiments.run_adapter import ADAPTER_DEFAULTS

    adapter = LSAdapter(d_ctx=120)
    opt = torch.optim.Adam(adapter.parameters(), lr=0.05)
    ctx = make_ctx_features().detach()
    w_c = torch.ones(4)
    # synthetic days: truth variance = 2x scenario variance -> ideal s = sqrt(2)
    days = []
    for _ in range(32):
        scen = torch.randn(50, 24, 4)
        y = torch.randn(24, 4) * np.sqrt(2.0)
        days.append((scen, y))
    for _ in range(n_steps):
        # the s-regularizer stays on: under MSE the data term carries NO scale
        # signal (the ensemble mean is s-invariant, gradient is pure float
        # roundoff that Adam would otherwise amplify), so lam_s keeps s at 1;
        # under CRPS the data term must overpower it and move s.
        loss = torch.stack([
            lsa_loss(adapter, scen, y, ctx, w_c,
                     lam_s=ADAPTER_DEFAULTS["lam_s"],
                     lam_delta=ADAPTER_DEFAULTS["lam_delta"],
                     objective=objective)
            for scen, y in days]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s_all = [adapter.compute_params(ctx, scen)[1] for scen, _ in days[:8]]
    return torch.stack(s_all).mean().item()


def test_scale_identifiable_under_crps():
    s = _train_scale("crps")
    assert 1.2 < s < 1.65, f"CRPS should push s toward sqrt(2)=1.414, got {s:.3f}"


def test_scale_not_identifiable_under_mse():
    # residual drift around 1 is Adam amplifying float-roundoff gradients (the
    # true MSE gradient of s is exactly 0); what matters is that s stays FAR
    # from the identifiable target sqrt(2)
    s = _train_scale("mse")
    assert abs(s - 1.0) < 0.1, f"MSE must leave s near 1 (not identifiable), got {s:.3f}"
    assert s < 1.15, f"MSE must not move s toward sqrt(2), got {s:.3f}"


# 4. ensemble CRPS matches properscoring (+ fair correction) ------------------
def test_ensemble_crps_formula():
    """properscoring implements the BIASED 1/(2M^2) estimator; ours is the fair
    1/(2M(M-1)) one, so the references differ exactly by t2/(2(M-1))."""
    ps = pytest.importorskip("properscoring")
    M = 64
    scen = torch.randn(M, 24, 4)
    y = torch.randn(24, 4)
    ours = ensemble_crps(scen, y).numpy()
    t2 = (scen.unsqueeze(0) - scen.unsqueeze(1)).abs().mean((0, 1)).numpy()
    ref = np.empty((24, 4))
    for h in range(24):
        for c in range(4):
            ref[h, c] = ps.crps_ensemble(y[h, c].item(), scen[:, h, c].numpy())
    ref_fair = ref - t2 / (2 * (M - 1))
    assert np.abs(ours - ref_fair).max() < 1e-5


# 5. no look-ahead leakage ------------------------------------------------------
def test_no_leakage(tmp_path):
    """metrics logging must precede context.push and any optimizer.step each day."""
    import pandas as pd

    from deployment import replay as replay_mod
    from deployment.replay import ReplayRunner

    # build a tiny fake cache + csv
    cache = tmp_path / "cache"
    cache.mkdir()
    dates = pd.date_range("2019-12-25", "2020-01-10", freq="D")
    for d in dates:
        np.savez(cache / f"{d.date()}.npz",
                 scenarios=np.random.randn(10, 24, 4).astype(np.float32) + 5,
                 y_true=np.random.randn(24, 4).astype(np.float32) + 5,
                 weather_future=np.zeros((24, 4), np.float32), date=str(d.date()))
    csv = tmp_path / "energy.csv"
    n = 43824 + 100
    df = pd.DataFrame({"date": pd.date_range("2014-01-01", periods=n, freq="h")})
    for col in ["PV", "Electricity", "Cooling", "Heat",
                "Temperature", "DewPoint", "Humidity", "GHI"]:
        df[col] = np.random.rand(n) * 10 + 1
    df.to_csv(csv, index=False)

    from utils.config import Config
    cfg = Config(data_csv=str(csv), data_name="energy", K=3, adapt_steps=2,
                 adapt_every=1, adapter_lr=1e-3, grad_clip=0.1,
                 pred_len=24, num_target=4)

    adapter = LSAdapter()
    events = []

    def loss_fn(a, scen, y, ctx):
        return lsa_loss(a, scen, y, ctx, torch.ones(4))

    runner = ReplayRunner(cfg, "proposed", adapter, loss_fn,
                          lambda c: c.features(), str(cache))

    orig_log = runner.logger.log
    orig_push = runner.context.push
    orig_step = runner.optimizer.step
    runner.logger.log = lambda *a, **k: (events.append("log"), orig_log(*a, **k))
    runner.context.push = lambda *a, **k: (events.append("push"), orig_push(*a, **k))
    runner.optimizer.step = lambda *a, **k: (events.append("step"), orig_step(*a, **k))

    warmup = [d for d in dates if d < pd.Timestamp("2020-01-01")]
    deploy = [d for d in dates if d >= pd.Timestamp("2020-01-01")]
    runner.prefill(warmup)
    events.clear()  # prefill pushes are legal (validation-period info)
    runner.run(deploy)

    # per day: exactly one log, one push, adapt_steps steps, strictly ordered
    per_day = len(deploy)
    assert events.count("log") == per_day
    assert events.count("push") == per_day
    day_chunks = []
    chunk = []
    for e in events:
        if e == "log" and chunk:
            day_chunks.append(chunk)
            chunk = []
        chunk.append(e)
    day_chunks.append(chunk)
    for chunk in day_chunks:
        assert chunk[0] == "log", "metrics must be recorded before anything else"
        assert chunk[1] == "push", "truth arrives only after logging"
        assert all(e == "step" for e in chunk[2:]), "updates strictly after push"
