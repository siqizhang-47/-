"""Config loading / seeding / device helpers for the low-carbon experiments."""
import hashlib
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml

DEFAULT_GPU = 0  # experiments run on GPU index 0 unless CUDA_VISIBLE_DEVICES already masks it


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path, common_path=None):
    """Load a model config, merged on top of the common config.

    If the model config contains an `inherit` key it is resolved relative to
    the config file's directory. `common_path` overrides that lookup.
    """
    cfg = load_yaml(config_path)
    base = {}
    inherit = cfg.pop("inherit", None)
    if common_path is None and inherit is not None:
        common_path = os.path.join(os.path.dirname(config_path), inherit)
    if common_path is not None and os.path.exists(common_path):
        base = load_yaml(common_path)
    merged = deep_update(base, cfg)
    merged["_config_path"] = os.path.abspath(config_path)
    merged["_config_hash"] = config_hash(merged)
    return merged


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def dict2ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict2ns(v) for k, v in d.items()})
    return d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = None, gpu: int = None) -> torch.device:
    """Resolve the compute device.

    Priority: explicit --device > explicit --gpu > CUDA_VISIBLE_DEVICES mask >
    default GPU index 6 > cpu.
    """
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu is None:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            # already masked by the launcher scripts (CUDA_VISIBLE_DEVICES=6)
            return torch.device("cuda:0")
        gpu = DEFAULT_GPU
    if gpu >= torch.cuda.device_count():
        gpu = 0
    return torch.device(f"cuda:{gpu}")


def rng_state_dict():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
