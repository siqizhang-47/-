"""TimeXer exogenous condition encoder (design document, sections 6-10).

Pipeline
--------
history_energy [B, 168, 4]
    -> TimeXer EnEmbedding (patch_len = stride = 24)   -> [B*4, 7+1, d]
    -> TimeXer Encoder (self-attention over patches,
       variate-wise cross-attention onto the exogenous tokens)
    -> target_tokens G' [B, 4, d]

future_calendar_raw [B, 24, 7] -> CyclicCalendarEncoder -> [B, 24, 11]
future_weather      [B, 24, 5]
    -> concat -> C_exo [B, 24, 16] -> variate tokenisation -> Z_exo [B, 16, d]

HorizonConditionAdapter then re-aligns the target-level condition to each of the
24 future hours and emits ``cond_horizon`` [B, 24, 4, d] and ``cond_denoiser``
[B, 24, 4*d].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data_provider.energy_weather_dataset import EXO_TOKEN_NAMES, EXO_TOKEN_TYPES
from ..third_party.timexer.SelfAttention_Family import AttentionLayer, FullAttention
from ..third_party.timexer.timexer_blocks import EnEmbedding, Encoder, EncoderLayer


class CyclicCalendarEncoder(nn.Module):
    """Raw calendar [B, H, 7] -> 11 encoded channels (design document, 5.3).

    Input channel order: Year, Month, DayOfYear, Hour, Weekday, IsWeekend, IsHoliday.
    Output order matches EXO_TOKEN_NAMES[:11].  The two binary flags pass through
    untouched -- they must never be standardised (24.10).
    """

    def __init__(self, year_start: int = 2014, year_span: int = 8, days_per_year: int = 365):
        super().__init__()
        self.year_start = year_start
        self.year_span = year_span
        self.days_per_year = days_per_year

    def forward(self, cal_raw: torch.Tensor) -> torch.Tensor:
        year = cal_raw[..., 0]
        month = cal_raw[..., 1]
        doy = cal_raw[..., 2]
        hour = cal_raw[..., 3]
        weekday = cal_raw[..., 4]
        is_weekend = cal_raw[..., 5]
        is_holiday = cal_raw[..., 6]

        two_pi = 2.0 * torch.pi
        year_norm = (year - self.year_start) / self.year_span
        month_ang = two_pi * (month - 1.0) / 12.0
        doy_ang = two_pi * (doy - 1.0) / self.days_per_year
        hour_ang = two_pi * hour / 24.0
        wd_ang = two_pi * weekday / 7.0

        return torch.stack([
            year_norm,
            torch.sin(month_ang), torch.cos(month_ang),
            torch.sin(doy_ang), torch.cos(doy_ang),
            torch.sin(hour_ang), torch.cos(hour_ang),
            torch.sin(wd_ang), torch.cos(wd_ang),
            is_weekend, is_holiday,
        ], dim=-1)


class ExogenousVariateTokenizer(nn.Module):
    """z_j = W_exo * C[:, j] + e_j^variable + e_j^type   (design document, section 8).

    Each exogenous variable contributes ONE token holding its whole 24-hour
    trajectory -- never a temporal average (24.4).
    """

    def __init__(self, n_exo: int, horizon: int, d_model: int, dropout: float = 0.1,
                 token_types=None):
        super().__init__()
        self.n_exo = n_exo
        self.value_projection = nn.Linear(horizon, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, n_exo, d_model) * 0.02)
        types = token_types if token_types is not None else EXO_TOKEN_TYPES[:n_exo]
        self.register_buffer("type_index", torch.tensor(types, dtype=torch.long))
        self.type_embedding = nn.Embedding(int(max(types)) + 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, c_exo: torch.Tensor) -> torch.Tensor:
        # c_exo: [B, H, n_exo] -> [B, n_exo, H] -> [B, n_exo, d]
        z = self.value_projection(c_exo.permute(0, 2, 1))
        z = z + self.variable_embedding + self.type_embedding(self.type_index).unsqueeze(0)
        return self.dropout(z)


class HorizonConditionAdapter(nn.Module):
    """Target-level condition -> per-hour, per-target condition (section 10).

        C_{h,k} = LayerNorm(C_target,k + E_h + R_h),   R_h = Linear(C_h^exo)

    ``cond_horizon``  [B, H, K, d]   feeds f_phi and g_psi
    ``cond_denoiser`` [B, H, K*d]    feeds the diffusion denoiser
    """

    def __init__(self, horizon: int, n_targets: int, n_exo: int, d_model: int,
                 use_exog: bool = True, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.use_exog = use_exog
        self.horizon_embedding = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.step_projection = nn.Linear(n_exo, d_model) if use_exog else None
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_tokens: torch.Tensor, c_exo: torch.Tensor | None):
        # target_tokens: [B, K, d]
        B, K, D = target_tokens.shape
        cond = target_tokens.unsqueeze(1).expand(B, self.horizon, K, D)
        e_h = self.horizon_embedding.view(1, self.horizon, 1, D)
        cond = cond + e_h
        if self.use_exog:
            r_h = self.step_projection(c_exo)              # [B, H, d]
            cond = cond + r_h.unsqueeze(2)
        cond_horizon = self.dropout(self.norm(cond))       # [B, H, K, d]
        cond_denoiser = cond_horizon.reshape(B, self.horizon, K * D)
        return cond_horizon, cond_denoiser


class TimeXerExogenousConditionEncoder(nn.Module):
    def __init__(self, window: int = 168, horizon: int = 24, n_targets: int = 4,
                 n_exo: int = 16, patch_len: int = 24, d_model: int = 128, n_heads: int = 8,
                 e_layers: int = 2, d_ff: int = 512, dropout: float = 0.1,
                 activation: str = "gelu", use_exog: bool = True,
                 shared_global_token: bool = False, factor: int = 3,
                 aux_history_branch: bool = False):
        super().__init__()
        assert window % patch_len == 0, "window must be an integer number of patches"
        self.window = window
        self.horizon = horizon
        self.n_targets = n_targets
        self.n_exo = n_exo
        self.d_model = d_model
        self.use_exog = use_exog
        self.shared_global_token = shared_global_token
        self.patch_num = window // patch_len
        # A5/A6 keep the exogenous condition away from some downstream modules.
        # Those modules must still receive *some* condition of the same shape,
        # otherwise the ablation would also be a capacity ablation.  The auxiliary
        # branch supplies a history-only condition built by a self-attention-only
        # copy of the endogenous encoder.
        self.aux_history_branch = bool(aux_history_branch and use_exog)

        self.calendar_encoder = CyclicCalendarEncoder()
        self.en_embedding = EnEmbedding(n_targets, d_model, patch_len, dropout,
                                        shared_glb_token=shared_global_token)
        if use_exog:
            self.exo_tokenizer = ExogenousVariateTokenizer(n_exo, horizon, d_model, dropout)
        else:
            self.exo_tokenizer = None

        def make_encoder(with_cross: bool):
            return Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, factor, attention_dropout=dropout,
                                          output_attention=False),
                            d_model, n_heads),
                        AttentionLayer(
                            FullAttention(False, factor, attention_dropout=dropout,
                                          output_attention=True),
                            d_model, n_heads) if with_cross else None,
                        d_model, d_ff, dropout=dropout, activation=activation,
                        n_vars=n_targets, shared_query=shared_global_token,
                    )
                    for _ in range(e_layers)
                ],
                norm_layer=nn.LayerNorm(d_model),
            )

        self.encoder = make_encoder(use_exog)
        self.horizon_adapter = HorizonConditionAdapter(
            horizon, n_targets, n_exo, d_model, use_exog=use_exog, dropout=dropout)

        if self.aux_history_branch:
            self.encoder_hist = make_encoder(False)
            self.horizon_adapter_hist = HorizonConditionAdapter(
                horizon, n_targets, n_exo, d_model, use_exog=False, dropout=dropout)

    # ------------------------------------------------------------------ helpers
    def build_exogenous(self, future_calendar_raw: torch.Tensor,
                        future_weather: torch.Tensor) -> torch.Tensor:
        """[B, H, 7] + [B, H, 5] -> C_exo [B, H, 16] in EXO_TOKEN_NAMES order."""
        cal = self.calendar_encoder(future_calendar_raw)
        return torch.cat([cal, future_weather], dim=-1)

    @property
    def token_names(self):
        return EXO_TOKEN_NAMES[:self.n_exo]

    # ------------------------------------------------------------------ forward
    def forward(self, history_energy: torch.Tensor, future_calendar_raw: torch.Tensor,
                future_weather: torch.Tensor):
        """history_energy [B, L, K]; future_calendar_raw [B, H, 7]; future_weather [B, H, 5]."""
        B = history_energy.shape[0]

        c_exo = None
        z_exo = None
        if self.use_exog:
            c_exo = self.build_exogenous(future_calendar_raw, future_weather)   # [B, H, 16]
            z_exo = self.exo_tokenizer(c_exo)                                   # [B, 16, d]

        en_embed, n_vars = self.en_embedding(history_energy.permute(0, 2, 1))   # [B*K, P+1, d]
        enc_out, attention = self.encoder(en_embed, z_exo)
        enc_out = enc_out.reshape(B, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        target_tokens = enc_out[:, :, -1, :]                                    # G' : [B, K, d]

        cond_horizon, cond_denoiser = self.horizon_adapter(target_tokens, c_exo)

        out = {
            "target_tokens": target_tokens,      # [B, 4, 128]
            "cond_horizon": cond_horizon,        # [B, 24, 4, 128]
            "cond_denoiser": cond_denoiser,      # [B, 24, 512]
            "attention": attention,              # [B, 8, 4, 16]  (None when use_exog=False)
            "patch_tokens": enc_out[:, :, :-1, :],
        }

        if self.aux_history_branch:
            hist_out, _ = self.encoder_hist(en_embed, None)
            hist_out = hist_out.reshape(B, n_vars, hist_out.shape[-2], hist_out.shape[-1])
            tokens_hist = hist_out[:, :, -1, :]
            cond_h_hist, cond_d_hist = self.horizon_adapter_hist(tokens_hist, None)
            out.update({
                "target_tokens_hist": tokens_hist,
                "cond_horizon_hist": cond_h_hist,
                "cond_denoiser_hist": cond_d_hist,
            })
        else:
            out.update({
                "target_tokens_hist": target_tokens,
                "cond_horizon_hist": cond_horizon,
                "cond_denoiser_hist": cond_denoiser,
            })
        return out
