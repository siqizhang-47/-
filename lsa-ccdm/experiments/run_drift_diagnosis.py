"""Exp 1: drift diagnosis (works directly on the cleaned csv, no GPU needed).

Usage: python experiments/run_drift_diagnosis.py [--n-boot 1000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, repo_root
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = repo_root()
    parser.add_argument("--base-config", default=str(root / "configs/base.yaml"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.base_config)
    set_seed(cfg.seed)

    from evaluation.drift_diagnosis import run_all
    run_all(root / cfg.data_csv, root / cfg.results_dir / "drift", n_boot=args.n_boot)


if __name__ == "__main__":
    main()
