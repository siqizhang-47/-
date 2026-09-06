"""TimeXer-inspired exogenous estimators for NsDiff.

Design goal
-----------
Only replace NsDiff's conditional mean estimator f_phi and, optionally, the
conditional variance estimator g_psi.  The NsDiff diffusion model, endpoint
construction, uncertainty-aware noise schedule, KL/noise/variance objectives,
and sampler are untouched.

The architecture is adapted from the official TimeXer implementation:
* endogenous targets -> non-overlapping patch tokens + one learnable global token;
* exogenous context -> one variate token per series;
* self-attention over endogenous patch/global tokens;
* exogenous-to-endogenous cross-attention through the global token;
* flatten projection head for multi-step prediction.

To retain the original NsDiff mean estimator's non-stationary mechanism, the
original tau/delta Projectors are retained.  DSAttention is used in place of
TimeXer's vanilla FullAttention.  Since delta is temporal (length T) while
TimeXer works at patch/variate granularity, it is mapped as follows:
* self-attention: average delta inside each patch and append its global mean;
* cross-attention: a learned linear projection maps temporal delta to one bias
  per exogenous variate token.
This keeps tau and delta trainable end-to-end under the SAME mean/variance loss
used by upstream NsDiff; no auxiliary objective is introduced.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.sigma import wv_sigma_trailing


class _OriginalSigmaEstimationBaseline(nn.Module):
    """Self-contained copy of upstream g_backbone.SigmaEstimation behavior.

    Keeping the baseline local avoids coupling this estimator module to unused
    torch_timeseries imports in g_backbone while preserving the exact rolling
    variance -> shared MLP -> Softplus computation.
    """

    def __init__(self, seq_len, pred_len, enc_in, hidden_size=512, kernel_size=24):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.kernel_size = kernel_size
        self.mlp = nn.Sequential(
            nn.Linear(seq_len - kernel_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        sigma = wv_sigma_trailing(x_enc, self.kernel_size, discard_rep=True)
        sigma = sigma[:, -(T - self.kernel_size):, :] + 1e-7
        pred_sigma = self.mlp(sigma.permute(0, 2, 1))
        pred_sigma = F.softplus(pred_sigma).permute(0, 2, 1)
        return pred_sigma[:, -self.pred_len:, :]


class DSAttention(nn.Module):
    """De-stationary attention copied conceptually from Non-stationary Transformer."""

    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # queries: [B,L,H,E], keys/values: [B,S,H,E]
        B, L, H, E = queries.shape
        S = keys.shape[1]
        scale = self.scale or 1.0 / math.sqrt(E)

        if tau is None:
            tau_ = 1.0
        else:
            # [B,1] or [B] -> [B,1,1,1]
            if tau.dim() == 1:
                tau = tau.unsqueeze(-1)
            tau_ = tau.unsqueeze(1).unsqueeze(1)

        if delta is None:
            delta_ = 0.0
        else:
            if delta.shape[-1] != S:
                raise ValueError(f"delta key length {delta.shape[-1]} != attention key length {S}")
            delta_ = delta.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau_ + delta_
        if self.mask_flag:
            if attn_mask is None:
                causal = torch.triu(torch.ones(L, S, device=scores.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal.view(1, 1, L, S), float("-inf"))
            else:
                mask = attn_mask.mask if hasattr(attn_mask, "mask") else attn_mask
                scores = scores.masked_fill(mask, float("-inf"))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous(), attn if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(q, k, v, attn_mask, tau=tau, delta=delta)
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


class Projector(nn.Module):
    """Original NsDiff/Non-stationary Transformer MLP for tau and delta."""

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x [B,S,E], stats [B,1,E]
        B = x.shape[0]
        x = self.series_conv(x)          # [B,1,E]
        x = torch.cat([x, stats], dim=1) # [B,2,E]
        return self.backbone(x.reshape(B, -1))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int):
        return self.pe[:, :length]


class EndogenousPatchEmbedding(nn.Module):
    """TimeXer endogenous embedding: patches + one learnable global token per target."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model) * 0.02)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [B,T,C]
        B, T, C = x.shape
        if C != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} endogenous vars, got {C}")
        if T % self.patch_len != 0:
            raise ValueError(f"Sequence length {T} must be divisible by patch_len={self.patch_len}")
        patches = x.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)  # [B,C,N,P]
        N = patches.shape[2]
        tokens = self.value_embedding(patches) + self.position_embedding(N).unsqueeze(1)
        glb = self.glb_token.expand(B, -1, -1, -1)
        tokens = torch.cat([tokens, glb], dim=2)  # [B,C,N+1,D]
        return self.dropout(tokens.reshape(B * C, N + 1, -1)), C


class ExogenousVariateEmbedding(nn.Module):
    """One token per target-history, weather, and time-feature series."""

    def __init__(self, seq_len, pred_len, n_targets, n_weather, n_time, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.full_len = seq_len + pred_len
        self.n_targets = n_targets
        self.n_weather = n_weather
        self.n_time = n_time
        self.target_proj = nn.Linear(seq_len, d_model, bias=False)
        self.weather_proj = nn.Linear(self.full_len, d_model, bias=False)
        self.time_proj = nn.Linear(self.full_len, d_model, bias=False)
        self.target_norm = nn.LayerNorm(d_model)
        self.weather_norm = nn.LayerNorm(d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def token_count(self):
        return self.n_targets + self.n_weather + self.n_time

    def forward(self, target_context, weather_hist, weather_future, time_hist, time_future):
        # target_context [B,T,Ct], weather [B,T/O,Cw], time [B,T/O,Ctime]
        target_tok = self.target_norm(self.target_proj(target_context.permute(0, 2, 1)))
        weather_full = torch.cat([weather_hist, weather_future], dim=1)
        weather_tok = self.weather_norm(self.weather_proj(weather_full.permute(0, 2, 1)))
        time_full = torch.cat([time_hist, time_future], dim=1)
        time_tok = self.time_norm(self.time_proj(time_full.permute(0, 2, 1)))
        return self.dropout(torch.cat([target_tok, weather_tok, time_tok], dim=1))


class TimeXerDSBlock(nn.Module):
    """TimeXer block with DSAttention and global-token-only cross attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.self_attention = AttentionLayer(
            DSAttention(False, attention_dropout=dropout, output_attention=False), d_model, n_heads
        )
        self.cross_attention = AttentionLayer(
            DSAttention(False, attention_dropout=dropout, output_attention=False), d_model, n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), act, nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x, context, tau=None, delta_self=None, delta_cross=None):
        sa, _ = self.self_attention(x, x, x, tau=tau, delta=delta_self)
        x = self.norm1(x + self.dropout(sa))

        glb = x[:, -1:, :]
        ca, _ = self.cross_attention(glb, context, context, tau=tau, delta=delta_cross)
        glb = self.norm2(glb + self.dropout(ca))
        x = torch.cat([x[:, :-1, :], glb], dim=1)

        y = self.ffn(x)
        return self.norm3(x + self.dropout(y))


class FlattenHead(nn.Module):
    def __init__(self, d_model, token_count, pred_len, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(d_model * token_count, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [B,C,L,D] -> [B,C,O]
        return self.dropout(self.linear(x.flatten(start_dim=-2)))


def _patch_delta(delta: torch.Tensor, patch_len: int) -> torch.Tensor:
    B, T = delta.shape
    if T % patch_len != 0:
        raise ValueError(f"delta length {T} not divisible by patch_len={patch_len}")
    patch = delta.reshape(B, T // patch_len, patch_len).mean(dim=-1)
    global_bias = delta.mean(dim=-1, keepdim=True)
    return torch.cat([patch, global_bias], dim=-1)


class TimeXerExogenousMean(nn.Module):
    """Replacement for f_phi(X), preserving the upstream MSE target and tau/delta."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int = 4,
        n_weather: int = 4,
        n_time: int = 4,
        patch_len: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.05,
        activation: str = "gelu",
        p_hidden_dims=(64, 64),
        p_hidden_layers: int = 2,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("Mean estimator requires seq_len divisible by patch_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len

        self.en_embedding = EndogenousPatchEmbedding(n_targets, d_model, patch_len, dropout)
        self.ex_embedding = ExogenousVariateEmbedding(
            seq_len, pred_len, n_targets, n_weather, n_time, d_model, dropout
        )
        self.encoder = nn.ModuleList(
            [TimeXerDSBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = FlattenHead(d_model, self.patch_num + 1, pred_len, dropout)

        # Exact NsDiff Non-stationary Transformer de-stationary factor learners.
        self.tau_learner = Projector(n_targets, seq_len, list(p_hidden_dims), p_hidden_layers, 1)
        self.delta_learner = Projector(n_targets, seq_len, list(p_hidden_dims), p_hidden_layers, seq_len)
        self.delta_to_context = nn.Linear(seq_len, self.ex_embedding.token_count, bias=False)

    def forward(self, target_hist, weather_hist, weather_future, time_hist, time_future):
        x_raw = target_hist.clone()
        mean = target_hist.mean(dim=1, keepdim=True).detach()
        centered = target_hist - mean
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = centered / stdev

        tau = self.tau_learner(x_raw, stdev).exp()  # [B,1]
        delta = self.delta_learner(x_raw, mean)      # [B,T]
        d_self = _patch_delta(delta, self.patch_len)
        d_cross = self.delta_to_context(delta)

        en, n_vars = self.en_embedding(x_norm)
        context = self.ex_embedding(x_norm, weather_hist, weather_future, time_hist, time_future)
        B = target_hist.shape[0]
        context = context.unsqueeze(1).expand(B, n_vars, -1, -1).reshape(B * n_vars, context.shape[1], context.shape[2])
        tau_r = tau.repeat_interleave(n_vars, dim=0)
        d_self_r = d_self.repeat_interleave(n_vars, dim=0)
        d_cross_r = d_cross.repeat_interleave(n_vars, dim=0)

        x = en
        for block in self.encoder:
            x = block(x, context, tau=tau_r, delta_self=d_self_r, delta_cross=d_cross_r)
        x = self.final_norm(x)
        x = x.reshape(B, n_vars, self.patch_num + 1, -1)
        out = self.head(x).permute(0, 2, 1)  # [B,O,C]
        out = out * stdev + mean
        aux: Dict[str, torch.Tensor] = {"tau": tau, "delta": delta}
        return out, aux


class TimeXerExogenousVariance(nn.Module):
    """TimeXer-style replacement for g_psi(X), with the SAME sqrt-variance MSE target.

    The endogenous branch receives the same trailing-window variance signal used by
    upstream g_backbone.SigmaEstimation.  Weather forecast and calendar features are
    supplied as exogenous variate tokens.  Positive output is enforced with Softplus,
    exactly as in the original variance estimator.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int = 4,
        n_weather: int = 4,
        n_time: int = 4,
        rolling_length: int = 96,
        patch_len: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.05,
        activation: str = "gelu",
        p_hidden_dims=(64, 64),
        p_hidden_layers: int = 2,
    ):
        super().__init__()
        sigma_len = seq_len - rolling_length
        if sigma_len <= 0:
            raise ValueError("rolling_length must be smaller than seq_len")
        if sigma_len % patch_len != 0:
            raise ValueError(
                f"Variance estimator sigma_len={sigma_len} must be divisible by patch_len={patch_len}. "
                "For seq_len=168 and rolling_length=96, patch_len=24 works."
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.rolling_length = rolling_length
        self.sigma_len = sigma_len
        self.patch_len = patch_len
        self.patch_num = sigma_len // patch_len

        self.en_embedding = EndogenousPatchEmbedding(n_targets, d_model, patch_len, dropout)
        self.ex_embedding = ExogenousVariateEmbedding(
            seq_len, pred_len, n_targets, n_weather, n_time, d_model, dropout
        )
        self.encoder = nn.ModuleList(
            [TimeXerDSBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = FlattenHead(d_model, self.patch_num + 1, pred_len, dropout)

        self.tau_learner = Projector(n_targets, sigma_len, list(p_hidden_dims), p_hidden_layers, 1)
        self.delta_learner = Projector(n_targets, sigma_len, list(p_hidden_dims), p_hidden_layers, sigma_len)
        self.delta_to_context = nn.Linear(sigma_len, self.ex_embedding.token_count, bias=False)

    def forward(self, target_hist, weather_hist, weather_future, time_hist, time_future):
        B, T, C = target_hist.shape
        sigma = wv_sigma_trailing(target_hist, self.rolling_length, discard_rep=True)
        # Match upstream g_backbone: retain exactly T - rolling_length points.
        sigma = sigma[:, -(T - self.rolling_length):, :] + 1e-8

        sigma_raw = sigma
        sigma_mean = sigma.mean(dim=1, keepdim=True).detach()
        centered = sigma - sigma_mean
        sigma_std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        sigma_norm = centered / sigma_std

        tau = self.tau_learner(sigma_raw, sigma_std).exp()
        delta = self.delta_learner(sigma_raw, sigma_mean)
        d_self = _patch_delta(delta, self.patch_len)
        d_cross = self.delta_to_context(delta)

        # Target-history context uses the same per-sample normalization as f_phi.
        t_mean = target_hist.mean(dim=1, keepdim=True).detach()
        t_center = target_hist - t_mean
        t_std = torch.sqrt(torch.var(t_center, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        target_context = t_center / t_std

        en, n_vars = self.en_embedding(sigma_norm)
        context = self.ex_embedding(target_context, weather_hist, weather_future, time_hist, time_future)
        context = context.unsqueeze(1).expand(B, n_vars, -1, -1).reshape(B * n_vars, context.shape[1], context.shape[2])
        tau_r = tau.repeat_interleave(n_vars, dim=0)
        d_self_r = d_self.repeat_interleave(n_vars, dim=0)
        d_cross_r = d_cross.repeat_interleave(n_vars, dim=0)

        x = en
        for block in self.encoder:
            x = block(x, context, tau=tau_r, delta_self=d_self_r, delta_cross=d_cross_r)
        x = self.final_norm(x)
        x = x.reshape(B, n_vars, self.patch_num + 1, -1)
        out = self.head(x).permute(0, 2, 1)
        # Restore the local variance scale, then enforce positivity as upstream g_psi does.
        out = out * sigma_std + sigma_mean
        return F.softplus(out)

class ResidualExogenousVarianceV2(nn.Module):
    """Residual, horizon-aware conditional variance estimator.

    V2 deliberately keeps the original NsDiff rolling-variance MLP as a stable
    absolute-scale baseline and lets a lightweight exogenous module learn only a
    bounded *log-variance correction*:

        g_v2 = g_base * exp(delta_log_var)

    where ``delta_log_var`` is bounded by ``max_log_correction``.  The final
    correction layer is zero-initialized, so at initialization V2 is exactly the
    original NsDiff variance estimator rather than a randomly initialized large
    Transformer.

    Main inductive biases:
    * historical volatility remains the primary signal;
    * rolling variance is encoded in log space with short non-overlapping patches;
    * future weather/calendar values remain aligned to each forecast horizon;
    * the already-predicted conditional mean can condition variance, but is detached
      so variance supervision does not backpropagate into f_phi;
    * positivity is guaranteed multiplicatively, without denormalize->Softplus.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int = 4,
        n_weather: int = 4,
        n_time: int = 4,
        rolling_length: int = 96,
        patch_len: int = 12,
        d_model: int = 96,
        n_heads: int = 4,
        d_ff: int = 192,
        dropout: float = 0.05,
        base_hidden: int = 512,
        max_log_correction: float = 1.0,
        weather_regime_len: int = 24,
    ):
        super().__init__()
        sigma_len = seq_len - rolling_length
        if sigma_len <= 0:
            raise ValueError("rolling_length must be smaller than seq_len")
        if sigma_len % patch_len != 0:
            raise ValueError(
                f"V2 sigma_len={sigma_len} must be divisible by patch_len={patch_len}. "
                "For seq_len=168 and rolling_length=96, patch_len=12 gives six patches."
            )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if max_log_correction <= 0:
            raise ValueError("max_log_correction must be positive")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.n_weather = n_weather
        self.n_time = n_time
        self.rolling_length = rolling_length
        self.sigma_len = sigma_len
        self.patch_len = patch_len
        self.patch_num = sigma_len // patch_len
        self.d_model = d_model
        self.max_log_correction = float(max_log_correction)
        self.weather_regime_len = int(weather_regime_len)

        # 1) Stable absolute-scale baseline: exactly the upstream NsDiff g_psi.
        self.base_g = _OriginalSigmaEstimationBaseline(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=n_targets,
            hidden_size=base_hidden,
            kernel_size=rolling_length,
        )

        # 2) Lightweight historical log-volatility memory.
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_pos = nn.Parameter(torch.randn(1, 1, self.patch_num, d_model) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(1, n_targets, 1, d_model) * 0.02)

        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hist_encoder = nn.TransformerEncoder(hist_layer, num_layers=1)
        self.hist_norm = nn.LayerNorm(d_model)

        # 3) Horizon-aligned future condition embedding.
        self.future_exog_proj = nn.Sequential(
            nn.Linear(n_weather + n_time, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.weather_hist_proj = nn.Sequential(
            nn.Linear(3 * n_weather, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.mean_proj = nn.Linear(1, d_model)
        self.base_var_proj = nn.Linear(1, d_model)
        self.horizon_embedding = nn.Parameter(torch.randn(1, 1, pred_len, d_model) * 0.02)

        # 4) Each target x horizon query reads its own target's volatility history.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 5) Bounded multiplicative correction in log-variance space.
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

        # Populated on every forward for optional diagnostics. These tensors are
        # detached so retaining them does not keep an autograd graph alive.
        self.last_base_var: Optional[torch.Tensor] = None
        self.last_delta_log_var: Optional[torch.Tensor] = None

    def _rolling_variance(self, target_hist: torch.Tensor) -> torch.Tensor:
        B, T, C = target_hist.shape
        if C != self.n_targets:
            raise ValueError(f"Expected {self.n_targets} target variables, got {C}")
        sigma = wv_sigma_trailing(target_hist, self.rolling_length, discard_rep=True)
        sigma = sigma[:, -(T - self.rolling_length):, :]
        if sigma.shape[1] != self.sigma_len:
            raise ValueError(f"Expected sigma length {self.sigma_len}, got {sigma.shape[1]}")
        return sigma.clamp_min(1e-8)

    def _encode_volatility(self, sigma: torch.Tensor) -> torch.Tensor:
        """[B,L,C] rolling variance -> [B*C,N_patch,D] volatility memory."""
        B, L, C = sigma.shape
        log_sigma = torch.log(sigma)
        mu = log_sigma.mean(dim=1, keepdim=True).detach()
        centered = log_sigma - mu
        std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        z = centered / std

        # [B,L,C] -> [B,C,N_patch,patch_len]
        patches = z.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len)
        tokens = self.patch_proj(patches)
        tokens = tokens + self.patch_pos + self.target_embedding
        tokens = tokens.reshape(B * C, self.patch_num, self.d_model)
        return self.hist_norm(self.hist_encoder(tokens))

    def _weather_regime(self, weather_hist: torch.Tensor) -> torch.Tensor:
        """Compact recent-weather regime summary: mean, std and last observation."""
        if weather_hist.shape[-1] != self.n_weather:
            raise ValueError(f"Expected {self.n_weather} weather variables, got {weather_hist.shape[-1]}")
        recent_len = min(self.weather_regime_len, weather_hist.shape[1])
        recent = weather_hist[:, -recent_len:, :]
        mean = recent.mean(dim=1)
        std = torch.sqrt(torch.var(recent, dim=1, unbiased=False) + 1e-5)
        last = recent[:, -1, :]
        return self.weather_hist_proj(torch.cat([mean, std, last], dim=-1))

    def forward(
        self,
        target_hist: torch.Tensor,
        weather_hist: torch.Tensor,
        weather_future: torch.Tensor,
        time_hist: torch.Tensor,
        time_future: torch.Tensor,
        mean_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del time_hist  # retained in the API to match the other estimator variants
        B, _, C = target_hist.shape
        if weather_future.shape[1] != self.pred_len or time_future.shape[1] != self.pred_len:
            raise ValueError("Future exogenous inputs must have pred_len time steps")

        # A) Original NsDiff baseline variance. This branch remains trainable.
        base_var = self.base_g(target_hist).clamp_min(1e-7)  # [B,O,C]

        # B) Historical volatility memory for each target variable.
        sigma_hist = self._rolling_variance(target_hist)
        hist_tokens = self._encode_volatility(sigma_hist)  # [B*C,N,D]

        # C) One query per future horizon. Future weather is NOT compressed across time.
        future_exog = torch.cat([weather_future, time_future], dim=-1)
        future_exog = self.future_exog_proj(future_exog)  # [B,O,D]
        future_exog = future_exog + self._weather_regime(weather_hist).unsqueeze(1)

        # Broadcasting produces [B,C,O,D]: target-specific + horizon-specific queries.
        q = future_exog.unsqueeze(1) + self.horizon_embedding + self.target_embedding

        # D) Condition variance on the predicted mean without sending gradients into f_phi.
        if mean_hat is not None:
            if mean_hat.shape != base_var.shape:
                raise ValueError(f"mean_hat shape {mean_hat.shape} must equal {base_var.shape}")
            m = mean_hat.detach().permute(0, 2, 1).unsqueeze(-1)  # [B,C,O,1]
            q = q + self.mean_proj(m)

        # E) Let queries know the stable baseline scale, but avoid a self-referential
        # gradient path through this conditioning feature.
        log_base = torch.log(base_var.detach()).permute(0, 2, 1).unsqueeze(-1)
        q = q + self.base_var_proj(log_base)
        q = q.reshape(B * C, self.pred_len, self.d_model)

        # F) Horizon queries attend to that target's historical volatility patches.
        context, _ = self.cross_attention(
            query=q,
            key=hist_tokens,
            value=hist_tokens,
            need_weights=False,
        )
        fused = self.cross_norm(q + self.dropout(context))

        # G) Bounded log-variance residual. Zero initialization => delta=0 initially.
        raw_delta = self.correction_head(fused).squeeze(-1)
        delta_log_var = self.max_log_correction * torch.tanh(raw_delta)
        delta_log_var = delta_log_var.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # H) Multiplicative correction guarantees positive variance without Softplus.
        out = (base_var * torch.exp(delta_log_var)).clamp_min(1e-7)

        self.last_base_var = base_var.detach()
        self.last_delta_log_var = delta_log_var.detach()
        return out



# =============================================================================
# V3: Seasonal baseline + TimeXer global branch + local dilated-TCN branch
#     + horizon-aligned future exogenous queries + residual conditional variance
# =============================================================================


def _inverse_softplus(value: float) -> float:
    value = float(value)
    return math.log(math.expm1(value)) if value < 20.0 else value


class SeasonalBaseline(nn.Module):
    """Periodic forecast baseline built from the historical window.

        mu_seasonal_h = alpha_h * X_{t-24+h} + (1 - alpha_h) * X_{t-168+h}

    ``gate="fixed"`` uses a constant ``daily_weight`` (0.7 by default);
    ``gate="learned"`` predicts ``alpha_h`` per target from the horizon-aligned
    future weather / calendar features.  The learned gate is initialised at the
    fixed weight so both modes start from the same baseline.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int,
        n_weather: int,
        n_time: int,
        gate: str = "fixed",
        daily_weight: float = 0.7,
        hidden: int = 64,
        daily_period: int = 24,
        weekly_period: int = 168,
    ):
        super().__init__()
        if gate not in {"fixed", "learned"}:
            raise ValueError("gate must be 'fixed' or 'learned'")
        if seq_len < daily_period:
            raise ValueError("seq_len must cover at least one daily period")
        h = torch.arange(pred_len)
        daily_idx = seq_len - daily_period + (h % daily_period)
        if seq_len >= weekly_period:
            weekly_idx = seq_len - weekly_period + (h % weekly_period)
        else:  # no weekly lag available -> degrade gracefully to the daily lag
            weekly_idx = daily_idx.clone()
        self.register_buffer("daily_idx", daily_idx, persistent=False)
        self.register_buffer("weekly_idx", weekly_idx, persistent=False)
        self.gate = gate
        self.daily_weight = float(daily_weight)
        if gate == "learned":
            self.gate_mlp = nn.Sequential(
                nn.Linear(n_weather + n_time, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_targets),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
            p = min(max(self.daily_weight, 1e-4), 1 - 1e-4)
            self.gate_bias = nn.Parameter(torch.full((1, 1, n_targets), math.log(p / (1 - p))))

    def forward(self, target_hist, weather_future, time_future):
        daily = target_hist[:, self.daily_idx, :]
        weekly = target_hist[:, self.weekly_idx, :]
        if self.gate == "fixed":
            alpha = torch.full_like(daily, self.daily_weight)
        else:
            cond = torch.cat([weather_future, time_future], dim=-1)
            alpha = torch.sigmoid(self.gate_mlp(cond) + self.gate_bias)
        return alpha * daily + (1.0 - alpha) * weekly, alpha


class _CausalDilatedBlock(nn.Module):
    """Conv1d -> GELU -> Dropout -> Conv1d -> residual (causal, dilated)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):  # [B,C,T]
        h = self.conv1(F.pad(x, (self.pad, 0)))
        h = self.dropout(self.act(h))
        h = self.conv2(F.pad(h, (self.pad, 0)))
        return self.norm(x + self.dropout(h))


class LocalDilatedTCN(nn.Module):
    """Local temporal branch: dilations 1->2->4->8->16, kernel 3 by default.

    [B,T,C] -> [B,T,D] point-wise memory that keeps hour-level dynamics which
    the 24 h TimeXer patches compress away.
    """

    def __init__(
        self,
        n_in: int,
        d_model: int,
        hidden: int = 128,
        kernel_size: int = 3,
        dilations=(1, 2, 4, 8, 16),
        dropout: float = 0.05,
    ):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_CausalDilatedBlock(hidden, kernel_size, int(d), dropout) for d in dilations]
        )
        self.out = nn.Linear(hidden, d_model)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * int(sum(int(d) for d in dilations))

    def forward(self, x):  # [B,T,C]
        h = self.inp(x.permute(0, 2, 1))
        for block in self.blocks:
            h = block(h)
        return self.out(h.permute(0, 2, 1))


class FutureHorizonQueryEmbedding(nn.Module):
    """One query token per forecast horizon: e_h = MLP([W_h, T_h]) + E_h."""

    def __init__(self, pred_len: int, n_weather: int, n_time: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_weather + n_time, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.horizon_embedding = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, weather_future, time_future):
        cond = torch.cat([weather_future, time_future], dim=-1)  # [B,O,Cw+Ct]
        return self.dropout(self.proj(cond) + self.horizon_embedding)


class HorizonDecoderLayer(nn.Module):
    """Pre-norm block: horizon self-attention -> cross-attention to history -> FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, memory):
        h = self.norm1(q)
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + self.dropout(sa)
        h = self.norm2(q)
        ca, _ = self.cross_attn(h, memory, memory, need_weights=False)
        q = q + self.dropout(ca)
        return q + self.dropout(self.ffn(self.norm3(q)))


class DaylightGate(nn.Module):
    """m_h = sigmoid(a * (GHI_h - b)) computed from the (scaled) future GHI."""

    def __init__(self, ghi_index: int, night_level: float, init_slope: float = 4.0, init_offset: float = 0.25):
        super().__init__()
        self.ghi_index = int(ghi_index)
        self.log_slope = nn.Parameter(torch.tensor(math.log(init_slope)))
        # Threshold slightly above the scaled "GHI = 0" level.
        self.threshold = nn.Parameter(torch.tensor(float(night_level) + float(init_offset)))

    def forward(self, weather_future):  # -> [B,O]
        ghi = weather_future[..., self.ghi_index]
        return torch.sigmoid(self.log_slope.exp() * (ghi - self.threshold))


class TimeXerExogenousMeanV3(nn.Module):
    """V3 conditional mean estimator.

        mu_h = SeasonalBase_h + Head_mu( Decoder( Q_h , [F_global ; F_local] ) )_h

    * ``F_global``: TimeXer patch/global tokens of every target (patch = 24 h,
      DSAttention with the retained tau/delta projectors, variate-token context);
    * ``F_local``: dilated-TCN point-wise memory of the 168 h history;
    * ``Q_h``: horizon-aligned future weather + calendar queries (24 tokens);
    * ``SeasonalBase``: 24 h / 168 h lag baseline (fixed or learned gate).

    All components can be switched off for the ablation ladder:
    ``use_seasonal`` / ``use_horizon_query`` / ``use_local_tcn``.  When
    ``use_horizon_query=False`` the original TimeXer FlattenHead is used, so
    V1 (= V0 + seasonal baseline) is reproduced exactly.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int = 4,
        n_weather: int = 4,
        n_time: int = 4,
        patch_len: int = 24,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.05,
        activation: str = "gelu",
        p_hidden_dims=(64, 64),
        p_hidden_layers: int = 2,
        use_seasonal: bool = True,
        use_horizon_query: bool = True,
        use_local_tcn: bool = True,
        seasonal_gate: str = "fixed",
        seasonal_daily_weight: float = 0.7,
        dec_layers: int = 2,
        tcn_hidden: int = 128,
        tcn_kernel: int = 3,
        tcn_dilations=(1, 2, 4, 8, 16),
        pv_daylight_gate: bool = False,
        pv_index: int = 1,
        ghi_index: int = 3,
        ghi_night_level: float = 0.0,
        pv_floor: float = 0.0,
    ):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("Mean estimator requires seq_len divisible by patch_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_targets = n_targets
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.d_model = d_model
        self.use_seasonal = bool(use_seasonal)
        self.use_horizon_query = bool(use_horizon_query)
        self.use_local_tcn = bool(use_local_tcn) and self.use_horizon_query
        self.pv_daylight_gate = bool(pv_daylight_gate)
        self.pv_index = int(pv_index)
        self.pv_floor = float(pv_floor)

        # --- Global TimeXer branch (identical to the V2 mean estimator) ---
        self.en_embedding = EndogenousPatchEmbedding(n_targets, d_model, patch_len, dropout)
        self.ex_embedding = ExogenousVariateEmbedding(
            seq_len, pred_len, n_targets, n_weather, n_time, d_model, dropout
        )
        self.encoder = nn.ModuleList(
            [TimeXerDSBlock(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.tau_learner = Projector(n_targets, seq_len, list(p_hidden_dims), p_hidden_layers, 1)
        self.delta_learner = Projector(n_targets, seq_len, list(p_hidden_dims), p_hidden_layers, seq_len)
        self.delta_to_context = nn.Linear(seq_len, self.ex_embedding.token_count, bias=False)

        # --- Seasonal baseline ---
        if self.use_seasonal:
            self.seasonal = SeasonalBaseline(
                seq_len, pred_len, n_targets, n_weather, n_time,
                gate=seasonal_gate, daily_weight=seasonal_daily_weight,
            )

        # --- Horizon-aligned decoder ---
        if self.use_horizon_query:
            self.future_query = FutureHorizonQueryEmbedding(pred_len, n_weather, n_time, d_model, dropout)
            self.global_target_embedding = nn.Parameter(torch.randn(1, n_targets, 1, d_model) * 0.02)
            self.memory_type_embedding = nn.Parameter(torch.randn(2, 1, 1, d_model) * 0.02)
            self.memory_norm = nn.LayerNorm(d_model)
            if self.use_local_tcn:
                self.local_tcn = LocalDilatedTCN(
                    n_targets, d_model, hidden=tcn_hidden, kernel_size=tcn_kernel,
                    dilations=tuple(tcn_dilations), dropout=dropout,
                )
                self.local_position = PositionalEmbedding(d_model, max_len=max(512, seq_len))
            self.decoder = nn.ModuleList(
                [HorizonDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(dec_layers)]
            )
            self.dec_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, n_targets)
            # Start close to the seasonal baseline (small residual at init).
            with torch.no_grad():
                self.head.weight.mul_(0.1)
                self.head.bias.zero_()
        else:
            self.head = FlattenHead(d_model, self.patch_num + 1, pred_len, dropout)

        if self.pv_daylight_gate:
            self.daylight_gate = DaylightGate(ghi_index, ghi_night_level)

    def _global_branch(self, x_norm, x_raw, mean, stdev, weather_hist, weather_future, time_hist, time_future):
        B = x_norm.shape[0]
        tau = self.tau_learner(x_raw, stdev).exp()
        delta = self.delta_learner(x_raw, mean)
        d_self = _patch_delta(delta, self.patch_len)
        d_cross = self.delta_to_context(delta)
        en, n_vars = self.en_embedding(x_norm)
        context = self.ex_embedding(x_norm, weather_hist, weather_future, time_hist, time_future)
        context = context.unsqueeze(1).expand(B, n_vars, -1, -1).reshape(B * n_vars, context.shape[1], context.shape[2])
        tau_r = tau.repeat_interleave(n_vars, dim=0)
        d_self_r = d_self.repeat_interleave(n_vars, dim=0)
        d_cross_r = d_cross.repeat_interleave(n_vars, dim=0)
        x = en
        for block in self.encoder:
            x = block(x, context, tau=tau_r, delta_self=d_self_r, delta_cross=d_cross_r)
        x = self.final_norm(x)
        return x.reshape(B, n_vars, self.patch_num + 1, -1), tau, delta

    def forward(self, target_hist, weather_hist, weather_future, time_hist, time_future):
        B, T, C = target_hist.shape
        x_raw = target_hist.clone()
        mean = target_hist.mean(dim=1, keepdim=True).detach()
        centered = target_hist - mean
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = centered / stdev

        glb, tau, delta = self._global_branch(
            x_norm, x_raw, mean, stdev, weather_hist, weather_future, time_hist, time_future
        )  # [B,C,N+1,D]

        aux: Dict[str, torch.Tensor] = {"tau": tau, "delta": delta}
        if self.use_horizon_query:
            g_mem = glb + self.global_target_embedding + self.memory_type_embedding[0]
            memory = [g_mem.reshape(B, C * (self.patch_num + 1), self.d_model)]
            if self.use_local_tcn:
                local = self.local_tcn(x_norm) + self.local_position(T) + self.memory_type_embedding[1, 0]
                memory.append(local)
            memory = self.memory_norm(torch.cat(memory, dim=1))
            q = self.future_query(weather_future, time_future)
            for layer in self.decoder:
                q = layer(q, memory)
            z = self.dec_norm(q)                      # [B,O,D]
            residual = self.head(z)                   # [B,O,C], window-normalised units
            aux["features"] = z
        else:
            residual = self.head(glb).permute(0, 2, 1)  # [B,O,C]

        if self.use_seasonal:
            base, alpha = self.seasonal(target_hist, weather_future, time_future)
            out = base + residual * stdev
            aux["seasonal_base"] = base
            aux["seasonal_alpha"] = alpha
        else:
            out = residual * stdev + mean

        if self.pv_daylight_gate:
            m = self.daylight_gate(weather_future)  # [B,O]
            pv = m * out[..., self.pv_index] + (1.0 - m) * self.pv_floor
            out = torch.cat(
                [out[..., : self.pv_index], pv.unsqueeze(-1), out[..., self.pv_index + 1 :]], dim=-1
            )
            aux["daylight_gate"] = m
        aux["residual"] = residual * stdev
        return out, aux


class ResidualConditionalVarianceV3(nn.Module):
    """Forecast-residual conditional variance estimator.

        sigma_h^2 = Softplus(z_h) + eps,   z = Head_sigma(CrossAttn(Q_h, F_hist), stopgrad(mu_h))

    It is trained with the heteroscedastic Gaussian NLL of the forecast residual
    ``r_h = Y_h - stopgrad(mu_h)`` (see experiment file), i.e. it learns *how
    large the forecast error should be* given history, aligned future weather,
    the horizon, and the predicted centre, instead of the 96 h rolling variance.

    Inputs used for the query of each (target, horizon):
    * horizon-aligned future weather / calendar,
    * predicted mean (detached),
    * optional detached decoder features of the V3 mean estimator,
    * horizon and target embeddings.
    History memory per target: log rolling-variance patches + last-24 h value patches.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_targets: int = 4,
        n_weather: int = 4,
        n_time: int = 4,
        rolling_length: int = 96,
        patch_len: int = 12,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.05,
        min_var: float = 1e-5,
        init_var: float = 0.1,
        recent_len: int = 48,
        mean_feature_dim: Optional[int] = None,
        pv_daylight_gate: bool = False,
        pv_index: int = 1,
        ghi_index: int = 3,
        ghi_night_level: float = 0.0,
        night_std_init: float = 0.05,
    ):
        super().__init__()
        sigma_len = seq_len - rolling_length
        if sigma_len <= 0 or sigma_len % patch_len != 0:
            raise ValueError("seq_len - rolling_length must be a positive multiple of patch_len")
        if recent_len % patch_len != 0 or recent_len > seq_len:
            raise ValueError("recent_len must be a multiple of patch_len and <= seq_len")
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_targets, self.n_weather = n_targets, n_weather
        self.rolling_length, self.patch_len = rolling_length, patch_len
        self.sigma_len, self.recent_len = sigma_len, recent_len
        self.d_model = d_model
        self.min_var = float(min_var)
        self.pv_daylight_gate = bool(pv_daylight_gate)
        self.pv_index = int(pv_index)

        n_sig = sigma_len // patch_len
        n_rec = recent_len // patch_len
        self.sig_patch_proj = nn.Linear(patch_len, d_model)
        self.rec_patch_proj = nn.Linear(patch_len, d_model)
        self.mem_pos = nn.Parameter(torch.randn(1, 1, n_sig + n_rec, d_model) * 0.02)
        self.target_embedding = nn.Parameter(torch.randn(1, n_targets, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.hist_encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.hist_norm = nn.LayerNorm(d_model)

        self.future_exog_proj = nn.Sequential(nn.Linear(n_weather + n_time, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.horizon_embedding = nn.Parameter(torch.randn(1, 1, pred_len, d_model) * 0.02)
        self.mean_proj = nn.Linear(1, d_model)
        self.mean_feature_proj = nn.Linear(mean_feature_dim, d_model) if mean_feature_dim else None
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, _inverse_softplus(init_var))

        if self.pv_daylight_gate:
            self.daylight_gate = DaylightGate(ghi_index, ghi_night_level)
            self.night_log_std = nn.Parameter(torch.tensor(math.log(night_std_init)))

    def _history_memory(self, target_hist):
        B, T, C = target_hist.shape
        sigma = wv_sigma_trailing(target_hist, self.rolling_length, discard_rep=True)
        sigma = sigma[:, -self.sigma_len:, :].clamp_min(1e-8)
        log_sigma = torch.log(sigma)
        mu = log_sigma.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(torch.var(log_sigma - mu, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        z_sig = (log_sigma - mu) / std
        sig_tokens = self.sig_patch_proj(z_sig.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len))

        recent = target_hist[:, -self.recent_len:, :]
        r_mean = recent.mean(dim=1, keepdim=True).detach()
        r_std = torch.sqrt(torch.var(recent - r_mean, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        z_rec = (recent - r_mean) / r_std
        rec_tokens = self.rec_patch_proj(z_rec.permute(0, 2, 1).unfold(-1, self.patch_len, self.patch_len))

        tokens = torch.cat([sig_tokens, rec_tokens], dim=2) + self.mem_pos + self.target_embedding  # [B,C,N,D]
        tokens = tokens.reshape(B * C, -1, self.d_model)
        return self.hist_norm(self.hist_encoder(tokens))

    def forward(
        self,
        target_hist,
        weather_hist,
        weather_future,
        time_hist,
        time_future,
        mean_hat: Optional[torch.Tensor] = None,
        mean_features: Optional[torch.Tensor] = None,
    ):
        del weather_hist, time_hist
        B, _, C = target_hist.shape
        if mean_hat is None:
            raise ValueError("ResidualConditionalVarianceV3 requires mean_hat")
        memory = self._history_memory(target_hist)  # [B*C,N,D]

        q = self.future_exog_proj(torch.cat([weather_future, time_future], dim=-1))  # [B,O,D]
        q = q.unsqueeze(1) + self.horizon_embedding + self.target_embedding          # [B,C,O,D]
        q = q + self.mean_proj(mean_hat.detach().permute(0, 2, 1).unsqueeze(-1))
        if self.mean_feature_proj is not None and mean_features is not None:
            q = q + self.mean_feature_proj(mean_features.detach()).unsqueeze(1)
        q = q.reshape(B * C, self.pred_len, self.d_model)

        ctx, _ = self.cross_attention(q, memory, memory, need_weights=False)
        fused = self.cross_norm(q + self.dropout(ctx))
        z = self.head(fused).squeeze(-1).reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B,O,C]
        var = F.softplus(z) + self.min_var

        if self.pv_daylight_gate:
            m = self.daylight_gate(weather_future)                       # [B,O]
            day_std = torch.sqrt(var[..., self.pv_index])
            night_std = self.night_log_std.exp()
            pv_std = m * day_std + (1.0 - m) * night_std
            pv_var = pv_std.square() + self.min_var
            var = torch.cat(
                [var[..., : self.pv_index], pv_var.unsqueeze(-1), var[..., self.pv_index + 1 :]], dim=-1
            )
        return var
