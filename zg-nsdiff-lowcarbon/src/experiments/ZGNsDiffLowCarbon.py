"""ZG-NsDiff full pipeline CLI: Stage F+Gate -> Stage G -> Joint -> test
(spec sections 8 & 9).

python -m src.experiments.ZGNsDiffLowCarbon \
    --config configs/zg_nsdiff_low_carbon.yaml --pretrain --seeds 1 2 3
"""
import argparse

from src.experiments.low_carbon_prob_forecast import LowCarbonNsDiffTrainer
from src.utils.config import get_device, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pretrain", action="store_true", default=False,
                    help="run Stage F+Gate and Stage G before joint training")
    ap.add_argument("--skip_train", action="store_true", default=False)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    variant = cfg.get("variant", "zg")
    device = get_device(args.device, args.gpu)
    print(f"device: {device}  variant: {variant}  model: {cfg.get('model_name', 'zg_nsdiff')}")
    for seed in args.seeds:
        print(f"===== {cfg.get('model_name', 'zg_nsdiff')}, seed {seed} =====")
        trainer = LowCarbonNsDiffTrainer(cfg, variant, seed, device)
        eff = {}
        if not args.skip_train:
            if args.pretrain:
                trainer.pretrain_f()
                trainer.pretrain_g()
            eff = trainer.train_joint(load_pretrain=args.pretrain)
        trainer.test_and_export(efficiency_extra=eff)


if __name__ == "__main__":
    main()
