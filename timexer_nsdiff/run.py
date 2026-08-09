#!/usr/bin/env python
"""Entry point: run one ablation over the configured seeds.

    python run.py --ablation A7                 # 3 seeds, GPU 2, full protocol
    python run.py --ablation A0 --seeds 1       # single seed
    python run.py --ablation A4 --gpu 0 --epochs_mean 2 --epochs_diffusion 2  # smoke test

Results land in  results/<ablation>/seed<k>/  and the aggregate over seeds in
results/<ablation>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from src.config import ABLATIONS, add_cli_arguments, load_config
from src.data_provider.energy_weather_dataset import HEEWData, SplitYears
from src.engine.evaluate import evaluate, export_attention, print_metrics
from src.engine.trainer import Trainer
from src.models.nsdiff_conditioned import TimeXerNsDiff
from src.utils.reproduce import count_parameters, reproducible, resolve_device


class Tee:
    """Mirror stdout into the run's log file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", buffering=1)

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.file.write(msg + "\n")

    def close(self):
        self.file.close()


def build_cli():
    p = argparse.ArgumentParser(description="TimeXer-conditioned NsDiff on the HEEW dataset")
    add_cli_arguments(p)
    p.add_argument("--stages", type=str, default="all",
                   help="comma separated subset of mean,scale,joint,diffusion,finetune")
    p.add_argument("--eval_only", action="store_true",
                   help="skip training and evaluate the checkpoints already in the run dir")
    return p


def resolve_stage_list(spec):
    if spec in (None, "", "all"):
        return ["mean", "scale", "joint", "diffusion", "finetune"]
    return [s.strip() for s in spec.split(",") if s.strip()]


def run_single_seed(cfg, data, seed, stages):
    cfg.seed = seed
    run_dir = os.path.join(cfg.output_dir, cfg.ablation, f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    log = Tee(os.path.join(run_dir, "train.log"))

    log("#" * 78)
    log(f"# ablation {cfg.ablation}: {ABLATIONS[cfg.ablation]['desc']}")
    log(f"# seed {seed} | device cuda:{cfg.gpu} | exog={cfg.use_exog} "
        f"shared_token={cfg.shared_global_token} "
        f"cond->(f={cfg.cond_to_mean}, g={cfg.cond_to_scale}, denoiser={cfg.cond_to_denoiser})")
    log("#" * 78)

    reproducible(seed, cfg.deterministic)
    device = resolve_device(cfg.gpu)

    model = TimeXerNsDiff(cfg).to(device)
    log(f"[model] trainable parameters: {count_parameters(model):,}")
    cfg.save(os.path.join(run_dir, "config.json"))

    train_set = data.dataset("train")
    val_set = data.dataset("val")
    test_set = data.dataset("test")

    trainer = Trainer(model, cfg, device, train_set, val_set, run_dir, logger=log)
    if not stages:
        ckpt = os.path.join(run_dir, "final_model.pth")
        if not os.path.exists(ckpt):
            log.close()
            sys.exit(f"[fatal] --eval_only needs a trained checkpoint at {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        log(f"[model] loaded {ckpt} for evaluation only")
    else:
        t0 = time.time()
        if "mean" in stages:
            trainer.train_mean()
        if "scale" in stages:
            trainer.train_scale()
        if "joint" in stages:
            trainer.train_joint()
        if "diffusion" in stages:
            trainer.train_diffusion()
        if "finetune" in stages:
            trainer.finetune_end_to_end()
        log(f"[timing] training wall clock: {(time.time() - t0) / 60:.1f} min")
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(trainer.history, f, indent=2)

    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))

    if cfg.save_attention and cfg.use_exog:
        export_attention(model, test_set, cfg, device, os.path.join(run_dir, "attention.json"))

    metrics = evaluate(model, test_set, cfg, device, data.target_scaler, run_dir, tag="test")
    print_metrics(metrics, logger=log)
    log.close()
    return metrics


def aggregate(all_metrics, cfg):
    """mean +/- std over the seeds for every scalar metric."""
    summary = {"ablation": cfg.ablation, "seeds": [m["seed"] for m in all_metrics],
               "weather_setting": "OracleWeather", "per_target": {}, "overall": {}}
    for target, m0 in all_metrics[0]["per_target"].items():
        summary["per_target"][target] = {}
        for key in m0:
            vals = [m["per_target"][target][key] for m in all_metrics]
            summary["per_target"][target][key] = {"mean": float(np.mean(vals)),
                                                  "std": float(np.std(vals))}
    for key, v0 in all_metrics[0]["overall"].items():
        if isinstance(v0, str):
            summary["overall"][key] = v0
            continue
        vals = [m["overall"][key] for m in all_metrics]
        summary["overall"][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out = os.path.join(cfg.output_dir, cfg.ablation, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] {cfg.ablation} over seeds {summary['seeds']}  ->  {out}")
    for target, m in summary["per_target"].items():
        print(f"  {target:<12} MAE={m['MAE']['mean']:.3f}+-{m['MAE']['std']:.3f}   "
              f"CRPS={m['CRPS']['mean']:.3f}+-{m['CRPS']['std']:.3f}   "
              f"PICP@90={m['PICP@90']['mean']:.3f}")
    return summary


def main():
    args = build_cli().parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "stages", "eval_only") and v is not None}
    cfg = load_config(args.config, overrides)
    cfg.apply_ablation(cfg.ablation)

    if not os.path.exists(cfg.data_path):
        sys.exit(f"[fatal] dataset not found: {cfg.data_path}\n"
                 f"        put the HEEW workbook there or pass --data_path <file.xlsx>")

    stages = [] if args.eval_only else resolve_stage_list(args.stages)

    data = HEEWData(cfg.data_path, window=cfg.window, horizon=cfg.horizon,
                    splits=SplitYears(tuple(cfg.train_years), tuple(cfg.val_years),
                                      tuple(cfg.test_years)),
                    cache_dir=cfg.cache_dir)

    all_metrics = []
    for seed in cfg.seeds:
        all_metrics.append(run_single_seed(cfg, data, seed, stages))
    if len(all_metrics) > 1:
        aggregate(all_metrics, cfg)


if __name__ == "__main__":
    main()
