"""Denoiser epsilon_theta / sigma_theta with a context that actually uses the
history embedding and the oracle future conditions (spec 7.5).

The original ConditionalGuidedModel received the history embedding but never
used it; this version concatenates a per-horizon context built from pooled
history and projected future conditions."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layer.exogenous_embedding import ExogenousDataEmbedding

EPS = 1e-8


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        return gamma.view(t.size(0), -1, self.num_out) * out


class ConditionalGuidedModelExo(nn.Module):
    def __init__(self, diff_steps, enc_in, condition_dim,
                 history_embed_dim=32, context_dim=64, hidden_dim=128, dropout=0.05):
        super().__init__()
        n_steps = diff_steps + 1
        self.history_embedding = ExogenousDataEmbedding(
            enc_in, condition_dim, history_embed_dim, dropout
        )
        self.history_pool = nn.Linear(history_embed_dim, context_dim)
        self.future_condition_projection = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        data_dim = enc_in * 3 + context_dim
        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = nn.Linear(hidden_dim, enc_in)
        self.sigma_lin = nn.Linear(hidden_dim, enc_in)

    def forward(self, history_target, history_condition, future_condition,
                y_t, y_0_hat, gx, t):
        history_emb = self.history_embedding(history_target, history_condition)  # [B,L,d]
        history_context = self.history_pool(history_emb.mean(dim=1))             # [B,d_ctx]
        future_context = self.future_condition_projection(future_condition)      # [B,H,d_ctx]
        context = history_context.unsqueeze(1) + future_context                  # [B,H,d_ctx]

        h = torch.cat([y_t, y_0_hat, gx, context], dim=-1)
        h = F.softplus(self.lin1(h, t))
        h = F.softplus(self.lin2(h, t))
        h = F.softplus(self.lin3(h, t))
        eps_pred = self.lin4(h)
        sigma = F.softplus(self.sigma_lin(F.softplus(h))) + EPS
        return eps_pred, sigma
