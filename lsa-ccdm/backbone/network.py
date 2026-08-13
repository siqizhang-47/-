"""Denoiser adapted from CCDM-main/network.py.

Changes vs. the source:
- the diffusion models ONLY the 4 target channels; weather enters as condition;
- new ``w_embedder`` for the future 24h weather covariates with two modes
  (config ``weather_cond_mode``):
    * ``token`` (default): weather tokens are appended to the DiT token
      sequence; the decoder keeps only the first ``num_target`` tokens;
    * ``pooled``: pooled weather embedding is added to the condition vector c;
- the timestamp condition (commented out in the source) is restored behind the
  ``use_time_cond`` switch.
DiTBlock itself (cross-channel attention over variable tokens) is unchanged.
"""
import torch
import torch.nn as nn

from backbone.attention import AttnMLP, FullAttention  # noqa: F401 (FullAttention used below)
from backbone.embed import DataEmbedding, StepEmbedding, TimeEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_dim, d_model, n_heads, attn_dropout, mlp_ratio=1.0, non_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        if non_attn:
            self.attn = nn.Linear(d_model, d_model, bias=True)
        else:
            self.attn = FullAttention(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = AttnMLP(in_dim=d_model, hidden_dim=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * d_model, bias=True)
        )
        self.non_attn = non_attn

    def forward(self, x, c):
        """
        x: (B, n_tokens, d_model)
        c: (B, hidden_dim)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod, x_mod, x_mod)
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x


class Decoder(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_dim, d_model, pred_len, n_emb):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            DataEmbedding(d_model, d_model, n_emb - 1),
            nn.Linear(d_model, pred_len)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * d_model, bias=True)
        )

    def forward(self, x, k):
        shift, scale = self.adaLN_modulation(k).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.mlp(x)
        return x  # (B, n_tokens, pred_len)


class Denoiser(nn.Module):
    def __init__(self, configs):
        super(Denoiser, self).__init__()
        self.num_feat = configs.num_feat        # history channels (8)
        self.num_target = configs.num_target    # diffusion channels (4)
        self.num_cov = self.num_feat - self.num_target
        self.weather_cond_mode = getattr(configs, "weather_cond_mode", "token")
        self.use_time_cond = getattr(configs, "use_time_cond", True)
        assert self.weather_cond_mode in ("token", "pooled")
        assert configs.step_hidden_dim == configs.time_hidden_dim, \
            "condition vector addition requires step_hidden_dim == time_hidden_dim"

        self.x_embedder = DataEmbedding(configs.cont_len, configs.cont_hidden_dim, configs.n_emb)
        self.y_embedder = DataEmbedding(configs.pred_len, configs.pred_hidden_dim, configs.n_emb)
        self.k_embedder = StepEmbedding(configs.step_hidden_dim, freq_dim=256)
        self.t_embedder = TimeEmbedding(configs.time_hidden_dim, configs.cont_len, configs.pred_len)

        d_model = configs.cont_hidden_dim + configs.pred_hidden_dim
        if self.weather_cond_mode == "token":
            self.w_embedder = DataEmbedding(configs.pred_len, configs.pred_hidden_dim, configs.n_emb)
            self.w_proj = nn.Linear(configs.pred_hidden_dim, d_model)
        else:
            self.w_embedder = DataEmbedding(configs.pred_len, configs.step_hidden_dim, configs.n_emb)
        if self.num_cov > 0:
            self.cov_proj = nn.Linear(configs.cont_hidden_dim, d_model)

        self.blocks = nn.ModuleList([
            DiTBlock(configs.step_hidden_dim, d_model, configs.n_heads, configs.attn_dropout,
                     configs.mlp_ratio, configs.non_attn)
            for _ in range(configs.n_depth)])
        self.decoder = Decoder(configs.step_hidden_dim, d_model, configs.pred_len, configs.n_emb)

    def forward(self, x, y, k, w=None, x_mark=None, y_mark=None):
        """
        x: (B, cont_len, num_feat)      history (targets + covariates)
        y: (B, pred_len, num_target)    noisy future targets
        k: (B, )                        diffusion step
        w: (B, pred_len, num_cov)       future weather condition
        returns (B, pred_len, num_target)
        """
        x_emb = self.x_embedder(x.permute(0, 2, 1))   # (B, num_feat, cont_hidden)
        y_emb = self.y_embedder(y.permute(0, 2, 1))   # (B, num_target, pred_hidden)
        c = self.k_embedder(k)                        # (B, step_hidden)
        if self.use_time_cond and x_mark is not None and y_mark is not None:
            c = c + self.t_embedder(x_mark, y_mark)

        # target tokens: history embedding of the target channel + noisy future embedding
        tokens = [torch.cat([x_emb[:, :self.num_target], y_emb], dim=-1)]  # (B, num_target, d_model)
        if self.num_cov > 0:
            tokens.append(self.cov_proj(x_emb[:, self.num_target:]))       # (B, num_cov, d_model)
        if w is not None:
            w_emb = self.w_embedder(w.permute(0, 2, 1))                     # (B, num_cov, *)
            if self.weather_cond_mode == "token":
                tokens.append(self.w_proj(w_emb))
            else:
                c = c + w_emb.mean(dim=1)

        h = torch.cat(tokens, dim=1)
        for block in self.blocks:
            h = block(h, c)
        out = self.decoder(h, c)                       # (B, n_tokens, pred_len)
        out = out[:, :self.num_target].permute(0, 2, 1)  # (B, pred_len, num_target)
        return out
