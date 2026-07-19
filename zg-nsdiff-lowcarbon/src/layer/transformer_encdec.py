"""Encoder/Decoder stacks. The Decoder exposes forward_hidden() so the
occurrence head can read the representation after the final LayerNorm but
before the target projection (spec 7.3)."""
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for layer in self.attn_layers:
            x, attn = layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # delta only modulates cross attention (official NS-Transformer behaviour)
        x = x + self.dropout(self.self_attention(x, x, x, x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, cross_mask, tau=tau, delta=delta)[0]
        )
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward_hidden(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """Hidden states after the final LayerNorm, before the projection."""
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        hidden = self.forward_hidden(x, cross, x_mask=x_mask, cross_mask=cross_mask,
                                     tau=tau, delta=delta)
        out = self.projection(hidden) if self.projection is not None else hidden
        return out, hidden
