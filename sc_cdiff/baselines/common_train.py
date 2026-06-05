"""Generic train / sample driver for any module exposing the SC-CDiff interface
(`loss(batch) -> (L, logs)` and `sample(batch, n) -> raw [B,n,5,24]`).

Reused by the SSSD and TimeGrad diffusion baselines so they share EMA and the
validation-Energy-Score early stopping with the main model."""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..artifacts import ckpt_path
from ..data.dataset import AlignedDays, TrainWindows
from ..ema import EMA
from ..eval.joint import mean_energy_score
from .common import collate, run_and_save


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _val_es(model, cfg, device, max_days, n_scen):
    ds = AlignedDays(cfg, "val", k=0)
    bs = cfg["sample"]["batch_days"]
    idx = list(range(min(len(ds), max_days)))
    scen_all, true_all = [], []
    for i in range(0, len(idx), bs):
        sub = [ds[j] for j in idx[i:i + bs]]
        batch = _move(collate(sub), device)
        scen_all.append(model.sample(batch, n_scen).detach().cpu().numpy())
        true_all.append(batch["Yraw"].cpu().numpy())
    return mean_energy_score(np.concatenate(scen_all, 0), np.concatenate(true_all, 0))


def train_module(cfg, model, device, desc="baseline"):
    model = model.to(device)
    tr = cfg["train"]
    ema = EMA(model, tr["ema_decay"])
    loader = DataLoader(TrainWindows(cfg), batch_size=tr["batch_size"], shuffle=True,
                        num_workers=tr["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    ckpt = ckpt_path(cfg)
    log = os.path.join(cfg["paths"]["artifacts"], f"train_log_{cfg['method_tag']}.csv")
    best, bad = float("inf"), 0
    with open(log, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "L", "val_es"])
        for epoch in range(tr["epochs"]):
            model.train(); tot = 0.0; nb = 0
            for batch in loader:
                batch = _move(batch, device)
                L, _ = model.loss(batch)
                opt.zero_grad(); L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
                opt.step(); ema.update(model)
                tot += float(L.item()); nb += 1
            val_es = ""
            if (epoch + 1) % tr["val_every"] == 0:
                if hasattr(ema.shadow, "norm"):
                    ema.shadow.norm = model.norm
                val_es = _val_es(ema.shadow.to(device), cfg, device,
                                 tr["val_subset_days"], tr["val_scenarios"])
                if val_es < best - 1e-4:
                    best, bad = val_es, 0
                    torch.save({"ema": ema.state_dict(), "cfg": cfg}, ckpt)
                    print(f"[{desc}] epoch {epoch} val_es={val_es:.4f} *saved*")
                else:
                    bad += 1
                    print(f"[{desc}] epoch {epoch} val_es={val_es:.4f} (best={best:.4f})")
            print(f"[{desc}] epoch {epoch} L={tot/max(nb,1):.4f}")
            w.writerow([epoch, tot / max(nb, 1), val_es]); fh.flush()
            if bad >= tr["patience"]:
                print(f"[{desc}] early stop"); break
    print(f"[{desc}] done best val_es={best:.4f} -> {ckpt}")


def sample_module(cfg, model, device, split="test", desc="baseline"):
    ckpt = torch.load(ckpt_path(cfg), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["ema"])
    model = model.to(device).eval()
    run_and_save(cfg, lambda batch, n: model.sample(batch, n), split=split,
                 device=device, desc=desc)
