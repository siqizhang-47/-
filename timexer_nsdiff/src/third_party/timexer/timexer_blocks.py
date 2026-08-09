"""TimeXer encoder blocks, taken from TimeXer-main/models/TimeXer.py.

``EnEmbedding``, ``Encoder`` and ``EncoderLayer`` are reproduced from the
original file.  Three additions were required to serve the design document and
the ablation table (section 20); every one of them is marked with ``# [MOD]``:

1. ``EnEmbedding`` can share one global endogenous token across all variables
   (ablation A3) instead of one token per variable (ablation A4).
2. ``EncoderLayer`` returns the variate-wise cross-attention map so that the
   ``[B, n_heads, 4, 15]`` tensor of section 9 can be inspected.
3. ``EncoderLayer`` accepts ``cross=None`` and then degenerates to a pure
   self-attention block, which is what ablation A0 (no exogenous variables)
   needs.

The original ``FlattenHead`` forecasting head is intentionally NOT vendored:
the design document states that TimeXer's prediction head is unused.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Embed import PositionalEmbedding


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout, shared_glb_token=False):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # [MOD] shared_glb_token=True -> a single global endogenous token is
        # reused by every target variable (ablation A3).
        self.shared_glb_token = shared_glb_token
        if shared_glb_token:
            self.glb_token = nn.Parameter(torch.randn(1, 1, 1, d_model))
        else:
            self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.n_vars = n_vars
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        if self.shared_glb_token:  # [MOD]
            glb = self.glb_token.repeat((x.shape[0], n_vars, 1, 1))
        else:
            glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        attn = None
        for layer in self.layers:
            # [MOD] keep the cross-attention map of the last layer
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", n_vars=1, shared_query=False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.n_vars = n_vars
        # [MOD] shared_query=True -> the four global tokens are pooled into a
        # single cross-attention query, so all targets receive the *same*
        # exogenous condition (ablation A3).
        self.shared_query = shared_query

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        attn = None
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        if cross is not None:  # [MOD] cross=None -> self-attention only (A0)
            B, L, D = cross.shape
            x_glb_ori = x[:, -1, :].unsqueeze(1)
            x_glb = torch.reshape(x_glb_ori, (B, -1, D))
            if self.shared_query:  # [MOD]
                n_q = x_glb.shape[1]
                x_glb_q = x_glb.mean(dim=1, keepdim=True)
                x_glb_attn, attn = self.cross_attention(
                    x_glb_q, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)
                x_glb_attn = x_glb_attn.expand(-1, n_q, -1)
            else:
                x_glb_attn, attn = self.cross_attention(
                    x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)
            x_glb_attn = self.dropout(x_glb_attn)
            x_glb_attn = torch.reshape(
                x_glb_attn,
                (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
            x_glb = x_glb_ori + x_glb_attn
            x_glb = self.norm2(x_glb)

            y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        else:
            y = x

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), attn
