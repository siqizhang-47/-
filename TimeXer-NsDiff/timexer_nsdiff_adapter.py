"""
TimeXer exogenous-condition encoder for NsDiff (implements the experiment plan).

Pipeline (§6-13 of the plan):
  history_energy [B,168,4] --EnEmbedding(patch)--> per-variable tokens
        --self-attention--> 4 target global tokens  G' [B,4,d]
  future calendar [B,24,5] --cyclic--> [B,24,9]  \
  future weather  [B,24,7] --standardised--------- concat -> exo [B,24,16]
        --variate tokenise--> 16 exogenous tokens Z_exo [B,16,d]
  cross-attention (Q=G', K/V=Z_exo) -> C_target [B,4,d] (+ attention [B,4,16])
        cond_global   = flatten(C_target)          [B, 4d = 512]
        cond_denoiser = horizon adapter            [B, 24, 4d = 512]

The endogenous tokenisation (EnEmbedding, global token) and the positional
embedding are taken from the original TimeXer code (reused/timexer_*).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from reused.timexer_embed import PositionalEmbedding

# fixed exogenous token order (for the attention heat-map, §23)
CALENDAR_TOKENS = ["YearTrend", "MonthSin", "MonthCos", "DaySin", "DayCos",
                   "HourSin", "HourCos", "WeekdaySin", "WeekdayCos"]          # 9
# NB: the dataset ships `Precip` in place of `Clearsky GHI`; kept as the 7th weather token.
WEATHER_TOKENS = ["Temperature", "DewPoint", "Humidity", "WindSpeed",
                  "WindGust", "Pressure", "Precip"]                            # 7
EXO_TOKENS = CALENDAR_TOKENS + WEATHER_TOKENS                                  # 16
N_CAL, N_WEATHER, N_EXO = 9, 7, 16
N_TARGETS = 4
TARGET_NAMES = ["Electricity", "PV", "Cooling", "Heat"]


class CyclicCalendarEncoder(nn.Module):
    """Raw calendar [B,H,5]=(Year,Month,Day,Hour,Weekday) -> [B,H,9] (§5.3)."""
    def forward(self, cal):
        year, month, day, hour, weekday = [cal[..., i] for i in range(5)]
        two_pi = 2 * math.pi
        feats = [
            (year - 2014.0) / 8.0,
            torch.sin(two_pi * (month - 1) / 12.0), torch.cos(two_pi * (month - 1) / 12.0),
            torch.sin(two_pi * (day - 1) / 31.0),   torch.cos(two_pi * (day - 1) / 31.0),
            torch.sin(two_pi * hour / 24.0),        torch.cos(two_pi * hour / 24.0),
            torch.sin(two_pi * weekday / 7.0),      torch.cos(two_pi * weekday / 7.0),
        ]
        return torch.stack(feats, dim=-1)


class EnEmbedding(nn.Module):
    """Endogenous patch embedding + per-variable global token (from TimeXer)."""
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                       # x: [B, n_vars, L]
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)     # [B,nv,np,pl]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)          # append global token
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class EndogenousEncoder(nn.Module):
    """History energy -> 4 target-specific global tokens G' [B,4,d] (§7)."""
    def __init__(self, n_vars, d_model, patch_len, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.n_vars = n_vars
        self.en_embed = EnEmbedding(n_vars, d_model, patch_len, dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)

    def forward(self, history_energy):          # [B, L, 4]
        x = history_energy.permute(0, 2, 1)     # [B, 4, L]
        tokens, n_vars = self.en_embed(x)       # [B*4, np+1, d]
        enc = self.encoder(tokens)              # self-attention over patches+glb
        glb = enc[:, -1, :]                     # global token per variable
        return glb.reshape(-1, n_vars, glb.shape[-1])   # [B, 4, d]


class ExogenousVariateTokenizer(nn.Module):
    """Future exo [B,H,16] -> 16 variate tokens Z_exo [B,16,d] (§8)."""
    def __init__(self, horizon, d_model, dropout):
        super().__init__()
        self.proj = nn.Linear(horizon, d_model)
        self.var_embed = nn.Embedding(N_EXO, d_model)          # variable identity
        self.type_embed = nn.Embedding(2, d_model)             # 0=calendar 1=weather
        type_ids = torch.tensor([0] * N_CAL + [1] * N_WEATHER)
        self.register_buffer("type_ids", type_ids)
        self.register_buffer("var_ids", torch.arange(N_EXO))
        self.dropout = nn.Dropout(dropout)

    def forward(self, exo):                     # [B, H, 16]
        z = self.proj(exo.transpose(1, 2))      # [B, 16, d]
        z = z + self.var_embed(self.var_ids)[None] + self.type_embed(self.type_ids)[None]
        return self.dropout(z)


class CrossAttentionConditioner(nn.Module):
    """Q=G' (4 targets), K/V=Z_exo (16) -> C_target [B,4,d] + attn [B,4,16] (§9)."""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, g_prime, z_exo):
        out, attn_w = self.attn(g_prime, z_exo, z_exo,
                                need_weights=True, average_attn_weights=False)  # [B,heads,4,16]
        c_target = self.norm(g_prime + out)
        return c_target, attn_w


class HorizonAdapter(nn.Module):
    """C_target [B,4,d] -> per-hour condition cond_denoiser [B,H,4d] (§10)."""
    def __init__(self, horizon, d_model):
        super().__init__()
        self.horizon = horizon
        self.horizon_embed = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.exo_proj = nn.Linear(N_EXO, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, c_target, exo):           # c_target[B,4,d], exo[B,H,16]
        B, K, d = c_target.shape
        R_h = self.exo_proj(exo)                                    # [B,H,d]
        base = c_target[:, None, :, :]                             # [B,1,4,d]
        eh = self.horizon_embed[None, :, None, :]                  # [1,H,1,d]
        rh = R_h[:, :, None, :]                                    # [B,H,1,d]
        c = self.norm(base + eh + rh)                             # [B,H,4,d]
        return c.reshape(B, self.horizon, K * d)                  # [B,H,4d]


class TimeXerExogenousConditionEncoder(nn.Module):
    """Full condition encoder -> dict(cond_global, cond_denoiser, attention)."""
    def __init__(self, horizon=24, d_model=128, patch_len=24, n_heads=8,
                 e_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.calendar = CyclicCalendarEncoder()
        self.endo = EndogenousEncoder(N_TARGETS, d_model, patch_len, n_heads, e_layers, d_ff, dropout)
        self.exo_tok = ExogenousVariateTokenizer(horizon, d_model, dropout)
        self.cross = CrossAttentionConditioner(d_model, n_heads, dropout)
        self.adapter = HorizonAdapter(horizon, d_model)
        self.d_model = d_model

    def forward(self, history_energy, future_calendar_raw, future_weather):
        cal = self.calendar(future_calendar_raw)                  # [B,H,9]
        exo = torch.cat([cal, future_weather], dim=-1)            # [B,H,16]
        g_prime = self.endo(history_energy)                       # [B,4,d]
        z_exo = self.exo_tok(exo)                                 # [B,16,d]
        c_target, attn = self.cross(g_prime, z_exo)              # [B,4,d],[B,heads,4,16]
        cond_global = c_target.reshape(c_target.shape[0], -1)     # [B,4d]
        cond_denoiser = self.adapter(c_target, exo)              # [B,H,4d]
        return {"cond_global": cond_global, "cond_denoiser": cond_denoiser,
                "attention": attn, "c_target": c_target}


class HistoryEncoder(nn.Module):
    """Encode past energy -> H_X [B, d_x] (GRU last hidden, §12)."""
    def __init__(self, n_vars=N_TARGETS, hidden=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_vars, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.hidden = hidden

    def forward(self, history_energy):
        _, h = self.gru(history_energy)
        return h[-1]                                              # [B, hidden]


class ConditionedLocationScaleHeads(nn.Module):
    """f_phi (mean) and g_psi (scale) on [H_X ; cond_global] (§12-13)."""
    def __init__(self, d_x, cond_dim, horizon=24, n_targets=N_TARGETS, hidden=256, eps=1e-4):
        super().__init__()
        self.horizon, self.n_targets, self.eps = horizon, n_targets, eps
        in_dim = d_x + cond_dim
        def mlp():
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, horizon * n_targets))
        self.mean_head = mlp()
        self.scale_head = mlp()

    def forward(self, history_feature, cond_global):
        h = torch.cat([history_feature, cond_global], dim=-1)
        B = h.shape[0]
        mu = self.mean_head(h).view(B, self.horizon, self.n_targets)
        sigma = F.softplus(self.scale_head(h)).view(B, self.horizon, self.n_targets) + self.eps
        return mu, sigma


def gaussian_location_scale_loss(y, mu, sigma):
    """Conditional Gaussian NLL (§13/§15.2)."""
    var = sigma ** 2
    return 0.5 * torch.mean((y - mu) ** 2 / var + torch.log(var))
