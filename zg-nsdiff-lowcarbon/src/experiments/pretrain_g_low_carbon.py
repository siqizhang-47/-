"""Stage G variance-network pretraining CLI."""
import argparse

from src.experiments.low_carbon_prob_forecast import LowCarbonNsDiffTrainer
from src.utils.config import get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device, args.gpu)
    print(f"device: {device}")
    for seed in args.seeds:
        print(f"===== pretrain G, seed {seed} =====")
        trainer = LowCarbonNsDiffTrainer(cfg, seed, device)
        trainer.pretrain_g()


if __name__ == "__main__":
    main()
