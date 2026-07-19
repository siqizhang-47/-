"""Embedding of target values + exogenous conditions (weather + calendar).

Replaces torch_timeseries DataEmbedding, whose x_mark slot only accepted the
fixed time-feature encoding (spec 7.2)."""
import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False
        )
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class ExogenousDataEmbedding(nn.Module):
    def __init__(self, value_dim: int, condition_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(value_dim, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, condition):
        return self.dropout(
            self.value_embedding(value)
            + self.position_embedding(value)
            + self.condition_projection(condition)
        )
