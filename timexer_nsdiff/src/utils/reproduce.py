"""Seeding and device helpers.

The experiment protocol uses exactly THREE random seeds (see configs/…yaml,
``seeds: [1, 2, 3]``); every reported number is the mean +/- std over those runs.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def reproducible(seed: int, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(gpu: int | str) -> torch.device:
    """`gpu=2` -> cuda:2 (the project default), `gpu=-1` or 'cpu' -> cpu."""
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    gpu = int(gpu)
    if gpu < 0 or not torch.cuda.is_available():
        if gpu >= 0:
            print("[device] CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu >= n:
        raise RuntimeError(f"requested cuda:{gpu} but only {n} GPU(s) are visible. "
                           f"If you used CUDA_VISIBLE_DEVICES, pass --gpu 0 instead.")
    return torch.device(f"cuda:{gpu}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
