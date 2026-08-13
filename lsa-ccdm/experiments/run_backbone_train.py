"""Phase 1: train the CCDM backbone (with early stopping + progress bars).

Usage:
    python experiments/run_backbone_train.py                       # wnorm on
    python experiments/run_backbone_train.py --override use_window_norm=false
    python experiments/run_backbone_train.py --extra-config configs/exp/smoke.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, repo_root, resolve_device
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--config", default=str(root / "configs/backbone_energy.yaml"))
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config, args.extra_config,
                      overrides=args.override)
    set_seed(cfg.seed)  # the ONE global seed
    cfg.device = resolve_device(cfg)

    from backbone.data_loader import create_mts_loader
    from backbone.model import DiffMTS

    tag = "wnorm_on" if cfg.use_window_norm else "wnorm_off"
    ckpt_path = str(root / cfg.checkpoint_dir / f"backbone_{tag}.pt")
    log_path = str(root / cfg.results_dir / f"backbone_{tag}_train_log.json")

    train_loader = create_mts_loader(cfg, "train")
    val_loader = create_mts_loader(cfg, "val", batch_size=cfg.train_batch_size, shuffle=False)

    model = DiffMTS(cfg, train_loader=train_loader, val_loader=val_loader)
    model.train(ckpt_path=ckpt_path, log_path=log_path)


if __name__ == "__main__":
    main()
