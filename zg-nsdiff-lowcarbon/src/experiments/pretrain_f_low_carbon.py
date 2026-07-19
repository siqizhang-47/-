"""Stage F (NsDiff) / Stage F+Gate (ZG-NsDiff) pretraining CLI (spec 9.1/9.2).

python -m src.experiments.pretrain_f_low_carbon \
    --config configs/nsdiff_low_carbon.yaml --seeds 1 2 3
"""
import argparse

from src.experiments.low_carbon_prob_forecast import LowCarbonNsDiffTrainer
from src.utils.config import get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = cfg.get("variant", "nsdiff")
    device = get_device(args.device, args.gpu)
    print(f"device: {device}  variant: {variant}")
    for seed in args.seeds:
        print(f"===== pretrain F, seed {seed} =====")
        trainer = LowCarbonNsDiffTrainer(cfg, variant, seed, device)
        trainer.pretrain_f()


if __name__ == "__main__":
    main()
