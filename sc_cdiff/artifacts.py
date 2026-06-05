"""Tag-namespaced artifact paths so SC-CDiff, its ablations, and every baseline
write side-by-side without clobbering each other.

tag comes from cfg['method_tag'] (default 'sccdiff')."""
from __future__ import annotations

import os


def _art(cfg):
    return cfg["paths"]["artifacts"]


def tag(cfg):
    return cfg.get("method_tag", "sccdiff")


def ckpt_path(cfg):
    return os.path.join(_art(cfg), f"ckpt_{tag(cfg)}.pt")


def ckpt_load_path(cfg):
    """Tag-namespaced checkpoint, falling back to the legacy ckpt_best.pt so
    older runs keep working after the artifact-naming change."""
    p = ckpt_path(cfg)
    if os.path.exists(p):
        return p
    legacy = os.path.join(_art(cfg), cfg["paths"].get("ckpt_best", "ckpt_best.pt"))
    return legacy if os.path.exists(legacy) else p


def scen_path(cfg, split, k, t=None):
    t = t or tag(cfg)
    return os.path.join(_art(cfg), f"scenarios_{t}_{split}_k{k}.npz")


def eval_path(cfg, split, k, t=None):
    t = t or tag(cfg)
    return os.path.join(_art(cfg), f"eval_{t}_{split}_k{k}.json")
