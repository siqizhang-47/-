"""Batch-run the SC-CDiff ablations: train + sample + evaluate each variant,
then print the master comparison table.

Variants (plan §7 / §8.4):
  sccdiff        full model
  wo_gate        --disable_gate
  wo_corr        --lambda_corr 0
  wo_era         --disable_era
  wo_cond        --disable_conditions
  oracle_weather full model conditioned on true weather (upper bound)

Run from the directory containing sc_cdiff/. Honors --device cuda:2.
"""
from __future__ import annotations

import argparse
import os

from .data.build_dataset import _load_cfg
from .eval.collect_results import collect
from .eval.run_eval import evaluate
from .sample import sample_split
from .train import pick_device

# (tag, ablation-dict, lambda_corr override or None, use_oracle_weather)
VARIANTS = [
    ("sccdiff", {}, None, False),
    ("wo_gate", {"disable_gate": True}, None, False),
    ("wo_corr", {}, 0.0, False),
    ("wo_era", {"disable_era": True}, None, False),
    ("wo_cond", {"disable_conditions": True}, None, False),
    ("oracle_weather", {}, None, True),
]


def _make_cfg(base_cfg_path, artifacts, device, epochs):
    cfg = _load_cfg(base_cfg_path)
    if artifacts:
        cfg["paths"]["artifacts"] = artifacts
    if device:
        cfg["device"] = device
    if epochs:
        cfg["train"]["epochs"] = epochs
    return cfg


def run_variant(base_cfg_path, tag, ablation, lam_corr, oracle,
                artifacts, device, epochs, split):
    cfg = _make_cfg(base_cfg_path, artifacts, device, epochs)
    cfg["method_tag"] = tag
    cfg.setdefault("ablation", {}).update(ablation)
    if lam_corr is not None:
        cfg["train"]["lambda_corr"] = lam_corr
    cfg["use_oracle_weather"] = oracle

    dev = pick_device(cfg["device"])
    print(f"\n########## VARIANT: {tag} ##########")
    _train_inline(cfg, dev)
    # sample + eval
    sample_split(cfg, dev, split=split)
    for k in cfg["sample"]["ks"]:
        evaluate(cfg, split, k)


def _train_inline(cfg, device):
    """Minimal training loop (mirrors train.py) parameterized by cfg."""
    import csv as _csv

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from .artifacts import ckpt_path
    from .data.dataset import TrainWindows
    from .ema import EMA
    from .models.sc_cdiff import SCCDiff
    from .normalize import Normalizer
    from .train import quick_val_es, set_seed

    set_seed(cfg["seed"])
    norm = Normalizer.load(os.path.join(cfg["paths"]["artifacts"], cfg["paths"]["norm_stats"]))
    model = SCCDiff(cfg, norm).to(device)
    ema = EMA(model, cfg["train"]["ema_decay"])
    loader = DataLoader(TrainWindows(cfg), batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], drop_last=True,
                        pin_memory=(device.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    best, bad = float("inf"), 0
    ckpt = ckpt_path(cfg)
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            L, _ = model.loss(batch)
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step(); ema.update(model)
        if (epoch + 1) % cfg["train"]["val_every"] == 0:
            ema.shadow.norm = norm
            es = quick_val_es(ema.shadow.to(device), cfg, device,
                              cfg["train"]["val_subset_days"], cfg["train"]["val_scenarios"])
            if es < best - 1e-4:
                best, bad = es, 0
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                            "cfg": cfg}, ckpt)
                print(f"[{cfg['method_tag']}] epoch {epoch} val_es={es:.4f} *saved*")
            else:
                bad += 1
            if bad >= cfg["train"]["patience"]:
                print(f"[{cfg['method_tag']}] early stop epoch {epoch}"); break
    if not os.path.exists(ckpt):  # ensure a checkpoint exists even without val improvement
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "cfg": cfg}, ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only", nargs="*", default=None, help="subset of variant tags to run")
    args = ap.parse_args()

    for tag, ab, lam, oracle in VARIANTS:
        if args.only and tag not in args.only:
            continue
        run_variant(args.config, tag, ab, lam, oracle,
                    args.artifacts, args.device, args.epochs, args.split)

    cfg = _make_cfg(args.config, args.artifacts, args.device, args.epochs)
    for k in cfg["sample"]["ks"]:
        collect(cfg, args.split, k)


if __name__ == "__main__":
    main()
