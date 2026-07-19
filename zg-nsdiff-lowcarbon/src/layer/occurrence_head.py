"""Occurrence (gate) head for zero-inflated targets (spec 8.1).

Reads the shared decoder hidden state (post final LayerNorm, pre projection)
and outputs logits ordered [cooling, heating, pv]."""
import torch.nn as nn


class OccurrenceHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, out_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, future_hidden):
        return self.net(future_hidden)
