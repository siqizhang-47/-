"""Experiment configuration + the ablation table of design-document section 20."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class ExpConfig:
    # ------------------------------------------------------------------ data
    data_path: str = "data/HEEW.xlsx"
    cache_dir: str = "data/cache"
    window: int = 168                 # L
    horizon: int = 24                 # H
    n_targets: int = 4
    n_exo: int = 16                   # 11 calendar + 5 weather (incl. ClearskyGHI)
    train_years: List[int] = field(default_factory=lambda: [2014, 2020])
    val_years: List[int] = field(default_factory=lambda: [2021, 2021])
    test_years: List[int] = field(default_factory=lambda: [2022, 2022])

    # ------------------------------------------------------------------ model
    patch_len: int = 24
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    activation: str = "gelu"
    head_hidden: int = 256
    head_layers: int = 2
    sigma_floor: float = 1e-4
    temporal_smoothing: bool = False
    denoiser_hidden: int = 128

    # ------------------------------------------------------------------ diffusion
    diffusion_steps: int = 20
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    rolling_length: int = 96
    num_samples: int = 100
    sample_chunk: int = 25

    # ------------------------------------------------------------------ ablation switches
    ablation: str = "A7"
    use_exog: bool = True
    shared_global_token: bool = False
    cond_to_mean: bool = True
    cond_to_scale: bool = True
    cond_to_denoiser: bool = True

    # ------------------------------------------------------------------ optimisation
    batch_size: int = 64
    eval_batch_size: int = 16
    num_workers: int = 4
    grad_clip: float = 1.0
    optimizer: str = "AdamW"

    epochs_mean: int = 50
    lr_mean: float = 1e-3
    wd_mean: float = 1e-4
    patience_mean: int = 8
    mean_loss: str = "mse"            # mse | huber

    epochs_scale: int = 30
    lr_scale_head: float = 5e-4
    lr_scale_encoder: float = 1e-4
    freeze_encoder_epochs: int = 5
    patience_scale: int = 8

    epochs_joint: int = 20
    lr_joint: float = 3e-4
    lambda_sigma: float = 0.1
    patience_joint: int = 8

    epochs_diffusion: int = 50
    lr_diffusion: float = 2e-4
    patience_diffusion: int = 8
    diff_val_metric: str = "loss"     # loss | crps
    val_subsample: int = 512          # windows used when diff_val_metric == crps

    epochs_finetune: int = 0          # stage 5 (optional end-to-end), 0 = skip
    lr_finetune: float = 2e-5
    patience_finetune: int = 5

    # ------------------------------------------------------------------ evaluation
    test_stride: int = 1
    variogram_mode: str = "cross_var"  # cross_var | full
    variogram_p: float = 0.5
    save_attention: bool = True
    save_samples: bool = False

    # ------------------------------------------------------------------ runtime
    gpu: int = 2
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    seed: int = 1
    output_dir: str = "results"
    run_name: str = ""
    deterministic: bool = False

    # ------------------------------------------------------------------ helpers
    def apply_ablation(self, name: str | None = None) -> "ExpConfig":
        name = (name or self.ablation).upper()
        if name not in ABLATIONS:
            raise ValueError(f"unknown ablation '{name}'. Available: {sorted(ABLATIONS)}")
        self.ablation = name
        for k, v in ABLATIONS[name]["switches"].items():
            setattr(self, k, v)
        return self

    def to_dict(self):
        return dataclasses.asdict(self)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------------------------------------------------------- ablations
ABLATIONS = {
    "A0": {
        "desc": "vanilla NsDiff: history of the four energy variables only, no exogenous input",
        "switches": dict(use_exog=False, shared_global_token=False,
                         cond_to_mean=False, cond_to_scale=False, cond_to_denoiser=False),
    },
    "A3": {
        "desc": "TimeXer with ONE shared global endogenous token (all targets share the exogenous condition)",
        "switches": dict(use_exog=True, shared_global_token=True,
                         cond_to_mean=True, cond_to_scale=True, cond_to_denoiser=True),
    },
    "A4": {
        "desc": "TimeXer with four target-specific global tokens (full model)",
        "switches": dict(use_exog=True, shared_global_token=False,
                         cond_to_mean=True, cond_to_scale=True, cond_to_denoiser=True),
    },
    "A5": {
        "desc": "A4 with the condition wired into f_phi only",
        "switches": dict(use_exog=True, shared_global_token=False,
                         cond_to_mean=True, cond_to_scale=False, cond_to_denoiser=False),
    },
    "A6": {
        "desc": "A4 with the condition wired into f_phi + g_psi",
        "switches": dict(use_exog=True, shared_global_token=False,
                         cond_to_mean=True, cond_to_scale=True, cond_to_denoiser=False),
    },
    "A7": {
        "desc": "A4 with the condition wired into f_phi + g_psi + denoiser (identical to A4)",
        "switches": dict(use_exog=True, shared_global_token=False,
                         cond_to_mean=True, cond_to_scale=True, cond_to_denoiser=True),
    },
}


def load_config(yaml_path: str | None = None, overrides: dict | None = None) -> ExpConfig:
    cfg = ExpConfig()
    if yaml_path:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        for k, v in flat.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"[config] ignoring unknown key '{k}'")
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if not hasattr(cfg, k):
                raise ValueError(f"unknown config field '{k}'")
            setattr(cfg, k, v)
    return cfg


def add_cli_arguments(parser: argparse.ArgumentParser):
    """Every ExpConfig field becomes an optional --flag that overrides the YAML."""
    parser.add_argument("--config", type=str, default="configs/energy_timexer_nsdiff.yaml")
    for f in dataclasses.fields(ExpConfig):
        if f.type == "bool" or f.type is bool:
            parser.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=None)
        elif f.name in ("seeds", "train_years", "val_years", "test_years"):
            parser.add_argument(f"--{f.name}", type=int, nargs="+", default=None)
        elif f.type in ("int", int):
            parser.add_argument(f"--{f.name}", type=int, default=None)
        elif f.type in ("float", float):
            parser.add_argument(f"--{f.name}", type=float, default=None)
        else:
            parser.add_argument(f"--{f.name}", type=str, default=None)
    return parser
