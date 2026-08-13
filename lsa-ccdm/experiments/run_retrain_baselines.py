"""Baselines 2/3: periodic retraining / yearly fine-tuning (v4 §16).

Every Jan 1 of a deployment year the backbone is retrained (or fine-tuned)
on ALL data up to the previous Dec 31, then that year's scenarios are sampled
into an independent cache. GPU hours are recorded for the cost axis (Fig. 7).

Usage:
    python experiments/run_retrain_baselines.py --mode retrain
    python experiments/run_retrain_baselines.py --mode finetune
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, repo_root, resolve_device
from utils.seed import set_seed

ROWS_PER_YEAR = {2014: 8760, 2015: 8760, 2016: 8784, 2017: 8760, 2018: 8760,
                 2019: 8760, 2020: 8784, 2021: 8760, 2022: 8760}


def rows_until(year_exclusive):
    return sum(ROWS_PER_YEAR[y] for y in range(2014, year_exclusive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--mode", choices=["retrain", "finetune"], required=True)
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--config", default=str(root / "configs/backbone_energy.yaml"))
    parser.add_argument("--deploy-config", default=str(root / "configs/deploy.yaml"))
    parser.add_argument("--extra-config", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.base_config, args.config, args.deploy_config,
                      args.extra_config, overrides=args.override)
    set_seed(cfg.seed)
    cfg.device = resolve_device(cfg)

    import torch

    from backbone.data_loader import create_mts_loader
    from backbone.model import DiffMTS
    from deployment.rolling import FrozenDeployer

    cost = {}
    for year in (2020, 2021, 2022):
        tag = f"{args.mode}_{year}"
        ckpt = root / cfg.checkpoint_dir / f"backbone_{tag}.pt"
        cache = root / cfg.scenario_dir / tag

        # data available on Jan 1 of `year`: everything up to Dec 31, year-1;
        # last available year serves as validation for early stopping
        train_end = rows_until(year - 1)
        val_end = rows_until(year)
        intervals = [train_end, val_end, val_end]

        t0 = time.time()
        if not ckpt.exists():
            train_loader = create_mts_loader(cfg, "train", intervals=intervals)
            val_loader = create_mts_loader(cfg, "val", intervals=intervals, shuffle=False)
            model = DiffMTS(cfg, train_loader=train_loader, val_loader=val_loader)
            if args.mode == "finetune":
                base_ckpt = root / cfg.checkpoint_dir / "backbone_wnorm_on.pt"
                model.load_weights(str(base_ckpt))
                model.denoiser.train()
                model.n_epochs = 5
                model.optimizer = torch.optim.Adam(
                    model.denoiser.parameters(), lr=cfg.init_lr * 0.1)
                model.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    model.optimizer, T_max=5)
                model.cfg.min_epochs = 1
                model.cfg.eval_every = 1
            model.train(ckpt_path=str(ckpt))
        train_seconds = time.time() - t0

        # deploy that single year with the fresh checkpoint
        t0 = time.time()
        deployer = FrozenDeployer(cfg, str(ckpt), intervals=intervals)
        deployer.run(cache, f"{year}-01-01", f"{year}-12-31", M=cfg.M,
                     warmup_days=0)
        cost[tag] = {"train_seconds": train_seconds,
                     "sample_seconds": time.time() - t0}

    cost_path = root / cfg.results_dir / f"{args.mode}_cost.json"
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_path, "w") as f:
        json.dump(cost, f, indent=2)
    print(f"Cost report written to {cost_path}")
    print("Evaluate with: python experiments/run_adapter.py --method frozen "
          f"--cache <scenario_dir>/{args.mode}_<year> --name {args.mode}_<year>")


if __name__ == "__main__":
    main()
