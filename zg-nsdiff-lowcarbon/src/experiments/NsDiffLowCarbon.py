"""NsDiff baseline joint training + single final test (spec sections 9 & 11).

python -m src.experiments.NsDiffLowCarbon \
    --config configs/nsdiff_low_carbon.yaml --load_pretrain --seeds 1 2 3
"""
import argparse

from src.experiments.low_carbon_prob_forecast import LowCarbonNsDiffTrainer
from src.utils.config import get_device, load_config


def run(config, seeds, load_pretrain, device=None, gpu=None, skip_train=False):
    cfg = load_config(config)
    variant = cfg.get("variant", "nsdiff")
    dev = get_device(device, gpu)
    print(f"device: {dev}  variant: {variant}  model: {cfg.get('model_name', variant)}")
    for seed in seeds:
        print(f"===== {cfg.get('model_name', variant)} joint, seed {seed} =====")
        trainer = LowCarbonNsDiffTrainer(cfg, variant, seed, dev)
        eff = {}
        if not skip_train:
            eff = trainer.train_joint(load_pretrain=load_pretrain)
        trainer.test_and_export(efficiency_extra=eff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--load_pretrain", action="store_true", default=False)
    ap.add_argument("--skip_train", action="store_true", default=False,
                    help="test/export only, from an existing best checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.seeds, args.load_pretrain, args.device, args.gpu, args.skip_train)


if __name__ == "__main__":
    main()
