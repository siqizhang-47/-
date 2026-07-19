"""De-stationary attention, minimal self-contained copy of the
Non-stationary Transformer components used by NsDiff's mu backbone."""
import math

import numpy as np
import torch
import torch.nn as nn


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(B, 1, L, L, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self):
        return self._mask


class DSAttention(nn.Module):
    """De-stationary attention: rescales scores with tau and shifts with delta."""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, Hh, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)      # B,1,1,1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B,1,1,S

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        Hh = self.n_heads
        queries = self.query_projection(queries).view(B, L, Hh, -1)
        keys = self.key_projection(keys).view(B, S, Hh, -1)
        values = self.value_projection(values).view(B, S, Hh, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
