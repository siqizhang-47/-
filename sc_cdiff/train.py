"""Step 11: training loop. AdamW + EMA, early stop on validation Energy Score."""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.build_dataset import _load_cfg
from .data.dataset import AlignedDays, TrainWindows
from .ema import EMA
from .eval.joint import mean_energy_score
from .models.sc_cdiff import SCCDiff, count_params
from .normalize import Normalizer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(want: str) -> torch.device:
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device(want)
    if want.startswith("cuda"):
        print(f"[train] {want} unavailable -> cpu")
    return torch.device("cpu")


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def quick_val_es(model, cfg, device, max_days, n_scen):
    """Day-ahead (k=0) Energy Score on a val subset using EMA weights."""
    ds = AlignedDays(cfg, "val", k=0)
    idx = list(range(min(len(ds), max_days)))
    bs = cfg["sample"]["batch_days"]
    scen_all, true_all = [], []
    for i in range(0, len(idx), bs):
        sub = [ds[j] for j in idx[i:i + bs]]
        batch = {k: torch.stack([s[k] for s in sub]).to(device) for k in sub[0]}
        scen = model.sample(batch, n_scen).cpu().numpy()     # [b,n,5,24]
        scen_all.append(scen)
        true_all.append(batch["Yraw"].cpu().numpy())
    scen = np.concatenate(scen_all, 0); truth = np.concatenate(true_all, 0)
    return mean_energy_score(scen, truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    if args.device:
        cfg["device"] = args.device
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    print(f"[train] device={device}")

    paths = cfg["paths"]
    norm = Normalizer.load(os.path.join(paths["artifacts"], paths["norm_stats"]))
    model = SCCDiff(cfg, norm).to(device)
    print(f"[train] params={count_params(model)/1e6:.2f}M")
    ema = EMA(model, cfg["train"]["ema_decay"])

    train_ds = TrainWindows(cfg)
    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    log_path = os.path.join(paths["artifacts"], paths["train_log"])
    ckpt_path = os.path.join(paths["artifacts"], paths["ckpt_best"])
    best_es, bad = float("inf"), 0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "L", "L_diff", "L_gate", "L_corr", "L_phy", "val_es"])

        for epoch in range(cfg["train"]["epochs"]):
            model.train()
            agg = {"L": 0, "L_diff": 0, "L_gate": 0, "L_corr": 0, "L_phy": 0}
            nb = 0
            for batch in loader:
                batch = move(batch, device)
                loss, logs = model.loss(batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
                ema.update(model)
                for k in agg:
                    agg[k] += logs[k]
                nb += 1
            for k in agg:
                agg[k] /= max(nb, 1)

            val_es = ""
            if (epoch + 1) % cfg["train"]["val_every"] == 0:
                ema.shadow.norm = norm
                val_es = quick_val_es(ema.shadow.to(device), cfg, device,
                                      cfg["train"]["val_subset_days"],
                                      cfg["train"]["val_scenarios"])
                improved = val_es < best_es - 1e-4
                if improved:
                    best_es, bad = val_es, 0
                    torch.save({"model": model.state_dict(),
                                "ema": ema.state_dict(), "cfg": cfg,
                                "val_es": val_es, "epoch": epoch}, ckpt_path)
                    print(f"[train] epoch {epoch} val_es={val_es:.4f} *saved*")
                else:
                    bad += 1
                    print(f"[train] epoch {epoch} val_es={val_es:.4f} (best={best_es:.4f}, bad={bad})")
            print(f"[train] epoch {epoch} L={agg['L']:.4f} L_diff={agg['L_diff']:.4f} "
                  f"L_gate={agg['L_gate']:.4f} L_corr={agg['L_corr']:.4f} L_phy={agg['L_phy']:.4f}")
            writer.writerow([epoch, agg["L"], agg["L_diff"], agg["L_gate"],
                             agg["L_corr"], agg["L_phy"], val_es])
            f.flush()
            if bad >= cfg["train"]["patience"]:
                print(f"[train] early stop at epoch {epoch}")
                break
    print(f"[train] done. best val_es={best_es:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
