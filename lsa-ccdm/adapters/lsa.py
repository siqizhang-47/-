"""Location-Scale Adapter (core new code, v4 §7-§9).

    (Delta, s) = g_phi(context)
    adapted^(m) = s_c * (Y^(m) - mu) + mu + Delta        (positive affine)

- Delta gated by tanh(g) with zero-initialized g  -> identity at start;
- Delta structure: lowrank (b_c + u_c @ Fourier basis) | fullrank | gated_only;
- s in [s_min, s_max] via sigmoid, initialized exactly at 1 (zero-init head);
- online objective: ensemble CRPS (proper scoring rule) + ridge regularizers.
"""
import math

import torch
import torch.nn as nn


def fourier_basis(H: int, n_basis: int) -> torch.Tensor:
    """(n_basis, H) fixed basis: [sin(2*pi*h/H), cos(2*pi*h/H), sin(4*pi*h/H), cos(4*pi*h/H), ...]"""
    h = torch.arange(H, dtype=torch.float32)
    rows = []
    freq = 1
    while len(rows) < n_basis:
        rows.append(torch.sin(2 * math.pi * freq * h / H))
        if len(rows) < n_basis:
            rows.append(torch.cos(2 * math.pi * freq * h / H))
        freq += 1
    return torch.stack(rows)


class LSAdapter(nn.Module):
    def __init__(self, H=24, C=4, d_ctx=120, d_hidden=64,
                 delta_mode="fullrank",   # 'lowrank' | 'fullrank' | 'gated_only' (ablation H)
                 n_basis=4,               # number of Fourier basis functions (lowrank)
                 scale_mode="carrier",    # 'carrier' | 'shared'                (ablation D)
                 s_min=0.5, s_max=2.0,
                 s_frozen=False, delta_frozen=False,
                 independent_scenarios=False,
                 use_pred_context=True):
        super().__init__()
        assert delta_mode in ("lowrank", "fullrank", "gated_only")
        assert scale_mode in ("carrier", "shared")
        self.H, self.C = H, C
        self.delta_mode = delta_mode
        self.scale_mode = scale_mode
        self.s_min, self.s_max = s_min, s_max
        self.s_frozen = s_frozen
        self.delta_frozen = delta_frozen
        self.independent_scenarios = independent_scenarios
        # today's predicted trajectory as adapter input (closes the structural
        # gap vs TAFAS/COSA whose corrections are functions of the prediction):
        # shape-normalized scenario mean (H*C) + raw per-carrier ensemble std (C).
        # No leakage: uses only the day's own frozen scenarios, never its truth.
        self.use_pred_context = use_pred_context
        self.d_pred = (H * C + C) if use_pred_context else 0

        self.mlp = nn.Sequential(
            nn.Linear(d_ctx + self.d_pred, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
        )

        # Delta head
        if delta_mode == "lowrank":
            self.delta_head = nn.Linear(d_hidden, C * (1 + n_basis))
            self.register_buffer("psi", fourier_basis(H, n_basis))  # (n_basis, H)
            self.n_basis = n_basis
        elif delta_mode == "fullrank":
            self.delta_head = nn.Linear(d_hidden, H * C)
        else:  # gated_only: per-carrier constant bias only
            self.delta_head = nn.Linear(d_hidden, C)
        # gating, zero-initialized -> start point IS the frozen backbone
        self.g = nn.Parameter(torch.zeros(C))

        # s head: zero-init weight + solved bias so that s == 1 exactly at init
        n_s = C if scale_mode == "carrier" else 1
        self.s_head = nn.Linear(d_hidden, n_s)
        nn.init.zeros_(self.s_head.weight)
        b0 = math.log((1.0 - s_min) / (s_max - 1.0))  # sigmoid(b0)=(1-s_min)/(s_max-s_min)
        nn.init.constant_(self.s_head.bias, b0)

        if s_frozen:
            self.s_head.weight.requires_grad_(False)
            self.s_head.bias.requires_grad_(False)
        if delta_frozen:
            self.delta_head.weight.requires_grad_(False)
            self.delta_head.bias.requires_grad_(False)
            self.g.requires_grad_(False)

    def _pred_features(self, scenarios):
        """(M, H, C) frozen scenarios -> (H*C + C,) prediction-trajectory features."""
        mu = scenarios.mean(0)                                     # (H, C)
        mu_shape = torch.nn.functional.layer_norm(
            mu.reshape(-1), (self.H * self.C,))                    # trajectory shape
        std_c = scenarios.std(0).mean(0)                           # dispersion level
        return torch.cat([mu_shape, std_c])

    def compute_params(self, ctx, scenarios=None):
        """ctx (d_ctx,), scenarios (M, H, C) -> Delta (H, C), s (C,)"""
        if self.use_pred_context:
            assert scenarios is not None, "use_pred_context needs the day's scenarios"
            ctx = torch.cat([ctx, self._pred_features(scenarios)])
        h = self.mlp(ctx)
        if self.delta_mode == "lowrank":
            out = self.delta_head(h).view(self.C, 1 + self.n_basis)
            b, u = out[:, 0], out[:, 1:]                      # (C,), (C, n_basis)
            delta_p = b[:, None] + u @ self.psi               # (C, H)
            delta_p = delta_p.t()                             # (H, C)
        elif self.delta_mode == "fullrank":
            delta_p = self.delta_head(h).view(self.H, self.C)
        else:
            delta_p = self.delta_head(h)[None, :].expand(self.H, self.C)
        delta = torch.tanh(self.g)[None, :] * delta_p         # (H, C)
        if self.delta_frozen:
            delta = torch.zeros_like(delta)

        s_raw = self.s_head(h)
        s = self.s_min + (self.s_max - self.s_min) * torch.sigmoid(s_raw)
        if self.scale_mode == "shared":
            s = s.expand(self.C)
        if self.s_frozen:
            s = torch.ones(self.C, device=ctx.device)
        return delta, s

    def forward(self, scenarios, ctx):
        """scenarios (M, H, C), ctx (d_ctx,) -> adapted scenarios (M, H, C)."""
        adapted, _, _ = self.forward_with_params(scenarios, ctx)
        return adapted

    def forward_with_params(self, scenarios, ctx):
        delta, s = self.compute_params(ctx, scenarios)
        mu = scenarios.mean(0, keepdim=True)
        adapted = s.view(1, 1, self.C) * (scenarios - mu) + mu + delta[None]
        return adapted, delta, s


def ensemble_crps(scen, y):
    """Fair (unbiased) sample CRPS, differentiable w.r.t. scenarios.

    Pair term uses 1/(2M(M-1)) instead of the naive 1/(2M^2).
    scen (M, H, C), y (H, C) -> per-(h, c) CRPS (H, C).
    """
    M = scen.shape[0]
    t1 = (scen - y.unsqueeze(0)).abs().mean(0)
    t2 = (scen.unsqueeze(0) - scen.unsqueeze(1)).abs().mean((0, 1))
    fair = M / (M - 1) if M > 1 else 1.0
    return t1 - 0.5 * fair * t2


def gaussian_nll(scen, y, eps=1e-6):
    """Gaussian-approximation NLL over the scenario ensemble (ablation I)."""
    mean = scen.mean(0)
    std = scen.std(0) + eps
    return torch.log(std) + (y - mean) ** 2 / (2 * std ** 2)


def lsa_loss(adapter, scenarios, y_true, ctx, w_c, lam_s=0.1, lam_delta=1e-3,
             objective="crps"):
    """Total online loss for one day (v4 §9.2). w_c: (C,) carrier weights."""
    adapted, delta, s = adapter.forward_with_params(scenarios, ctx)
    if objective == "crps":
        data_term = (w_c[None, :] * ensemble_crps(adapted, y_true)).mean()
    elif objective == "nll":
        data_term = (w_c[None, :] * gaussian_nll(adapted, y_true)).mean()
    elif objective == "mse":
        # deliberately wrong objective for ablation I: the ensemble mean is
        # invariant to s, so the scale is NOT identifiable under MSE
        data_term = ((w_c[None, :] * (adapted.mean(0) - y_true) ** 2)).mean()
    elif objective == "mse_per_scenario":
        # ablation J ("independent per-scenario adaptation" control): every
        # scenario is fitted to the truth independently. This drives s toward
        # s_min (dispersion contraction) and collapses calibration -- the
        # analytically predictable sanity-check outcome. (The plan's literal
        # mu^(m)=self formulation makes the scale term inert, so we implement
        # the spirit of the control via the per-scenario objective.)
        data_term = ((w_c[None, None, :] * (adapted - y_true.unsqueeze(0)) ** 2)).mean()
    else:
        raise NotImplementedError(objective)
    # both regularizers use MEAN reduction, matching the data term's mean over
    # the (h, c) cells -- a sum over carriers would inflate the s-penalty 4x
    # relative to the CRPS dispersion incentive and pin s at 1
    reg = lam_s * ((s - 1.0) ** 2).mean() + lam_delta * (delta ** 2).mean()
    return data_term + reg
