"""GAN baselines (plan §8.4): WGAN (unconditional, reference-paper family) and
conditional WGAN-GP. Compact 1D-conv generator/critic over the 5x24 day, with
gradient penalty. No structural gate -- only non-negativity at sampling."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..artifacts import ckpt_path
from ..data.build_dataset import _load_cfg
from ..data.dataset import TrainWindows
from ..normalize import Normalizer
from .common import run_and_save


class CondEnc(nn.Module):
    def __init__(self, cfg, d):
        super().__init__()
        self.conv = nn.Conv1d(cfg["n_weather"] + cfg["n_calendar"] + cfg["n_channels"], d, 3, padding=1)
        self.era = nn.Embedding(3, d)

    def forward(self, batch, conditional):
        x = torch.cat([batch["W"], batch["CAL"], batch["Yhat"]], dim=1)
        h = self.conv(x) + self.era(batch["era"]).unsqueeze(-1)
        if not conditional:
            h = torch.zeros_like(h)
        return h                                              # [B,d,24]


class Generator(nn.Module):
    def __init__(self, cfg, zdim=64, d=128):
        super().__init__()
        self.zdim = zdim
        self.enc = CondEnc(cfg, d)
        self.z_proj = nn.Linear(zdim, d)
        self.net = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1), nn.GELU(),
            nn.Conv1d(d, d, 3, padding=1), nn.GELU(),
            nn.Conv1d(d, cfg["n_channels"], 3, padding=1))

    def forward(self, z, batch, conditional):
        h = self.enc(batch, conditional)                      # [B,d,24]
        h = h + self.z_proj(z).unsqueeze(-1)
        return self.net(h)                                    # [B,5,24] standardized


class Critic(nn.Module):
    def __init__(self, cfg, d=128):
        super().__init__()
        self.enc = CondEnc(cfg, d)
        self.net = nn.Sequential(
            nn.Conv1d(cfg["n_channels"] + d, d, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(d, d, 3, padding=1), nn.LeakyReLU(0.2))
        self.head = nn.Linear(d, 1)

    def forward(self, x, batch, conditional):
        h = self.enc(batch, conditional)
        y = self.net(torch.cat([x, h], dim=1))
        return self.head(y.mean(dim=-1))                      # [B,1]


def _gp(critic, real, fake, batch, conditional, dev):
    a = torch.rand(real.size(0), 1, 1, device=dev)
    inter = (a * real + (1 - a) * fake).requires_grad_(True)
    out = critic(inter, batch, conditional)
    g = torch.autograd.grad(out, inter, torch.ones_like(out),
                            create_graph=True, retain_graph=True)[0]
    return ((g.reshape(g.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def train_gan(cfg, device, conditional):
    norm = Normalizer.load(os.path.join(cfg["paths"]["artifacts"], cfg["paths"]["norm_stats"]))
    G, Dc = Generator(cfg).to(device), Critic(cfg).to(device)
    optG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9))
    optD = torch.optim.Adam(Dc.parameters(), 1e-4, betas=(0.0, 0.9))
    loader = DataLoader(TrainWindows(cfg), batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=cfg["train"]["num_workers"], drop_last=True)
    n_critic, lam = 5, 10.0
    for epoch in range(cfg["train"]["epochs"]):
        gl = dl = 0.0; nb = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            real = batch["Y"]; B = real.size(0)
            for _ in range(n_critic):
                z = torch.randn(B, G.zdim, device=device)
                fake = G(z, batch, conditional).detach()
                lossD = (Dc(fake, batch, conditional).mean() - Dc(real, batch, conditional).mean()
                         + lam * _gp(Dc, real, fake, batch, conditional, device))
                optD.zero_grad(); lossD.backward(); optD.step()
            z = torch.randn(B, G.zdim, device=device)
            lossG = -Dc(G(z, batch, conditional), batch, conditional).mean()
            optG.zero_grad(); lossG.backward(); optG.step()
            gl += float(lossG.item()); dl += float(lossD.item()); nb += 1
        print(f"[{cfg['method_tag']}] epoch {epoch} lossG={gl/max(nb,1):.3f} lossD={dl/max(nb,1):.3f}")
    torch.save({"G": G.state_dict(), "conditional": conditional, "cfg": cfg}, ckpt_path(cfg))
    print(f"[{cfg['method_tag']}] saved -> {ckpt_path(cfg)}")


@torch.no_grad()
def sample_gan(cfg, device, split):
    norm = Normalizer.load(os.path.join(cfg["paths"]["artifacts"], cfg["paths"]["norm_stats"]))
    ck = torch.load(ckpt_path(cfg), map_location=device, weights_only=False)
    conditional = ck["conditional"]
    G = Generator(cfg).to(device); G.load_state_dict(ck["G"]); G.eval()

    def gen(batch, n):
        rep = lambda x: x.repeat_interleave(n, dim=0)
        rb = {k: (rep(v) if torch.is_tensor(v) else v) for k, v in batch.items()}
        z = torch.randn(rb["Y"].size(0), G.zdim, device=device)
        Ys = G(z, rb, conditional)
        Yr = torch.relu(norm.denormalize_torch(Ys, rb["era"]))
        Yr[:, cfg["pv_idx"]] = torch.clamp(Yr[:, cfg["pv_idx"]], 0, cfg["pv_cap"])
        B = batch["Y"].size(0)
        return Yr.reshape(B, n, 5, 24)
    run_and_save(cfg, gen, split=split, device=device, desc=cfg["method_tag"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--method", choices=["wgan", "cwgan_gp"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage", choices=["train", "sample", "both"], default="both")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    if args.device:
        cfg["device"] = args.device
    cfg["method_tag"] = args.method
    from ..train import pick_device
    device = pick_device(cfg["device"])
    conditional = (args.method == "cwgan_gp")
    if args.stage in ("train", "both"):
        train_gan(cfg, device, conditional)
    if args.stage in ("sample", "both"):
        sample_gan(cfg, device, args.split)


if __name__ == "__main__":
    main()
