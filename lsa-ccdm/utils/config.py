"""YAML config loading with `key=value` command-line overrides."""
import argparse
from pathlib import Path

import yaml


class Config(argparse.Namespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return dict(vars(self))


def _parse_value(raw: str):
    return yaml.safe_load(raw)


def load_config(*yaml_paths, overrides=None) -> Config:
    """Merge one or more yaml files (later files win), then apply overrides.

    overrides: iterable of "key=value" strings, values parsed as yaml.
    """
    merged = {}
    for path in yaml_paths:
        if path is None:
            continue
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping")
        merged.update(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        merged[key.strip()] = _parse_value(raw)
    return Config(**merged)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_device(cfg) -> str:
    import torch

    device = getattr(cfg, "device", "cuda:2")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {device} unavailable, falling back to cpu")
        return "cpu"
    return device
