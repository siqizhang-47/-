"""PositionalEmbedding copied verbatim from TimeXer-main/layers/Embed.py.

TimeXer's ``EnEmbedding`` uses this module to add intra-variate patch positions.
The other embeddings of the original file (DataEmbedding_inverted, TokenEmbedding,
TemporalEmbedding, ...) are not needed here: the exogenous branch of this project
uses its own variate tokenizer with variable-identity and variable-type
embeddings (see the design document, section 8).
"""
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
