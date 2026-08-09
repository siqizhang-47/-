"""NsDiff denoiser, taken from NsDiff-main/src/layer/denoise.py.

``ConditionalLinear`` is verbatim.  ``ConditionalGuidedModel`` keeps the exact
same 3-layer softplus trunk plus the twin (eps, sigma) output heads; the only
change (marked ``# [MOD]``) is the extra ``cond_dim`` input channel that carries
``cond_denoiser`` ([B, 24, 512]) from the TimeXer condition encoder, exactly as
prescribed by section 14 of the design document ("first version: plain
concatenation").  With ``cond_dim=0`` the module is bit-for-bit the original
NsDiff denoiser (data_dim = enc_in * 3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, diff_steps, enc_in, cond_dim=0, hidden=128):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = diff_steps + 1

        self.cond_dim = cond_dim  # [MOD]
        data_dim = enc_in * 3 + cond_dim  # [MOD] + external condition channels
        self.lin1 = ConditionalLinear(data_dim, hidden, n_steps)
        self.lin2 = ConditionalLinear(hidden, hidden, n_steps)
        self.lin3 = ConditionalLinear(hidden, hidden, n_steps)
        self.lin4 = nn.Linear(hidden, enc_in)
        self.sigma_lin = nn.Linear(hidden, enc_in)

    def forward(self, y_t, y_0_hat, g_x, t, cond=None):
        # return eps_pred : noise    (B, O, N)
        #        sigma    : variance (B, O, N)
        parts = [y_t, y_0_hat, g_x]
        if self.cond_dim > 0:  # [MOD]
            assert cond is not None, "denoiser was built with cond_dim > 0 but no condition was given"
            parts.append(cond)
        eps_pred = torch.cat(parts, dim=-1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred, sigma = self.lin4(eps_pred), F.softplus(self.sigma_lin(F.softplus(eps_pred)))
        return eps_pred, sigma
