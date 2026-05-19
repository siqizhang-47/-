"""Construct a baseline model from its original repo source code.

Each baseline ships its own `models/`, `layers/`, `utils/` packages with
overlapping module names. We can't ``import`` them all in the same Python
process safely, so a single training run picks one baseline and prepends its
directory to ``sys.path`` before constructing the model. Use one
``python -m experiment.run --model <name>`` invocation per baseline.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn


MODEL_NAMES = ("iTransformer", "TimeMixer", "TimeXer")


_BASELINE_DIRS = {
    "iTransformer": "baselines/iTransformer-main",
    "TimeMixer": "baselines/TimeMixer-main",
    "TimeXer": "baselines/TimeXer-main",
}


def _prepend_path(p: str) -> None:
    abs_p = os.path.abspath(p)
    if abs_p not in sys.path:
        sys.path.insert(0, abs_p)


def _default_configs(name: str, *, seq_len: int, pred_len: int, enc_in: int) -> SimpleNamespace:
    """Reasonable hyper-params shared across the three Time-Series-Library style models.

    All three use ``features='MS'`` so the target column lives at the last
    index of the channel axis (see ``window.feature_columns``).
    """
    common = dict(
        task_name="long_term_forecast",
        features="MS",
        seq_len=seq_len,
        label_len=seq_len // 2,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        factor=1,
        dropout=0.1,
        embed="timeF",
        freq="t",
        activation="gelu",
        output_attention=False,
        use_norm=True,
        class_strategy="projection",
    )
    if name == "iTransformer":
        return SimpleNamespace(**common)
    if name == "TimeMixer":
        cfg = dict(common)
        cfg.update(
            moving_avg=25,
            down_sampling_layers=2,
            down_sampling_window=2,
            down_sampling_method="avg",
            channel_independence=1,
            decomp_method="moving_avg",
            use_future_temporal_feature=0,
            use_norm=1,
            top_k=5,
        )
        return SimpleNamespace(**cfg)
    if name == "TimeXer":
        cfg = dict(common)
        cfg.update(patch_len=16, use_norm=True)
        return SimpleNamespace(**cfg)
    raise ValueError(f"unknown model: {name}")


def build_model(
    name: str,
    *,
    seq_len: int,
    pred_len: int,
    enc_in: int,
    overrides: Dict | None = None,
) -> nn.Module:
    if name not in _BASELINE_DIRS:
        raise ValueError(f"model must be one of {MODEL_NAMES}, got {name}")
    _prepend_path(_BASELINE_DIRS[name])

    cfg = _default_configs(name, seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)

    if name == "iTransformer":
        from model.iTransformer import Model
    elif name == "TimeMixer":
        from models.TimeMixer import Model
    elif name == "TimeXer":
        from models.TimeXer import Model
    else:
        raise AssertionError("unreachable")
    return Model(cfg)
