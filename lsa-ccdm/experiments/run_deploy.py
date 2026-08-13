"""Phase 2: frozen daily deployment + scenario caching (run ONCE per backbone).

Usage:
    python experiments/run_deploy.py                                # wnorm_on
    python experiments/run_deploy.py --override use_window_norm=false
    python experiments/run_deploy.py --extra-config configs/exp/smoke.yaml  # Jan 2020, M=20
Resume is automatic: existing npz days are skipped.
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
    parser.add_argument("--deploy-config", default=str(root / "configs/deploy.yaml"))
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--ckpt", default=None, help="checkpoint path override")
    parser.add_argument("--out", default=None, help="cache dir override")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config, args.deploy_config,
                      args.extra_config, overrides=args.override)
    set_seed(cfg.seed)
    cfg.device = resolve_device(cfg)

    from deployment.rolling import FrozenDeployer

    tag = "wnorm_on" if cfg.use_window_norm else "wnorm_off"
    ckpt = args.ckpt or str(root / cfg.checkpoint_dir / f"backbone_{tag}.pt")
    out_dir = args.out or str(root / cfg.scenario_dir / tag)

    deployer = FrozenDeployer(cfg, ckpt)
    deployer.run(out_dir, cfg.deploy_start, cfg.deploy_end, M=cfg.M,
                 warmup_days=cfg.warmup_days)


if __name__ == "__main__":
    main()
