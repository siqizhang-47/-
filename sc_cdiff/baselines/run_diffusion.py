"""Train + sample the diffusion baselines that share the generic trainer:
SSSD (state-space) and TimeGrad (autoregressive)."""
from __future__ import annotations

import argparse
import os

from ..data.build_dataset import _load_cfg
from ..normalize import Normalizer
from ..train import pick_device
from .common_train import sample_module, train_module
from .diffusion_base import build_sssd
from .timegrad import build_timegrad

BUILDERS = {"sssd": build_sssd, "timegrad": build_timegrad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", choices=list(BUILDERS), required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--stage", choices=["train", "sample", "both"], default="both")
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    if args.artifacts:
        cfg["paths"]["artifacts"] = args.artifacts
    if args.device:
        cfg["device"] = args.device
    cfg["method_tag"] = args.model
    device = pick_device(cfg["device"])
    norm = Normalizer.load(os.path.join(cfg["paths"]["artifacts"], cfg["paths"]["norm_stats"]))
    model = BUILDERS[args.model](cfg, norm)
    if args.stage in ("train", "both"):
        train_module(cfg, model, device, desc=args.model)
    if args.stage in ("sample", "both"):
        sample_module(cfg, model, device, split=args.split, desc=args.model)


if __name__ == "__main__":
    main()
