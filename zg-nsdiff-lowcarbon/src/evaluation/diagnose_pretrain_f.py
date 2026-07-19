"""Diagnose Stage-F (mean network) pretraining quality in PHYSICAL units.

Converts the pretrained mu back to kW on the validation split, reports
per-variable MAE / RMSE / model-space MSE, and compares against a
persistence-24h naive baseline (copy the last 24 history hours).

Rule of thumb:
  * F clearly better than naive on electricity  -> pretraining is healthy,
    the joint stage will keep improving it;
  * F worse than or equal to naive              -> investigate (conditioning
    bug / lr / epochs) before running the joint stage.

Usage:
    PYTHONPATH=. python -m src.evaluation.diagnose_pretrain_f \
        --config configs/nsdiff_low_carbon.yaml --seed 1
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from src.data.low_carbon_schema import TARGET_NAMES
from src.experiments.low_carbon_prob_forecast import LowCarbonNsDiffTrainer, to_device
from src.utils.config import get_device, load_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", default=None,
                    help="default: artifacts/runs/<model>/seed_<k>/pretrain_f.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--max_batches", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device, args.gpu)
    trainer = LowCarbonNsDiffTrainer(cfg, cfg.get("variant", "nsdiff"), args.seed, device)
    ckpt = args.checkpoint or os.path.join(trainer.run_dir, "pretrain_f.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")
    trainer.load_checkpoint(ckpt, modules=("mean_model", "occurrence_head"))
    trainer.model.eval()
    tf = trainer.transform

    D = len(TARGET_NAMES)
    abs_f = np.zeros(D); sq_f = np.zeros(D)
    abs_n = np.zeros(D); sq_n = np.zeros(D)
    mse_model_space = np.zeros(D); n_model_space = np.zeros(D)
    n = 0
    H = trainer.model.H

    for bi, batch in enumerate(tqdm(trainer.dm.val_loader(), desc="diagnose F", ncols=110)):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        batch = to_device(batch, device)
        mu, _ = trainer.model.forward_mean(batch)

        # model-space per-variable masked MSE (what the printed val loss averages)
        mask = batch["future_observed"].float()
        err2 = ((mu - batch["future_target"]) ** 2 * mask).sum(dim=(0, 1)).cpu().numpy()
        mse_model_space += err2
        n_model_space += mask.sum(dim=(0, 1)).cpu().numpy()

        # physical units
        mu_raw = tf.inverse_torch(mu.cpu(), target_dim=-1).numpy()
        truth = batch["future_target_raw"].cpu().numpy()
        naive = batch["history_target_raw"][:, -H:, :].cpu().numpy()  # persistence-24h
        abs_f += np.abs(mu_raw - truth).sum(axis=(0, 1))
        sq_f += np.square(mu_raw - truth).sum(axis=(0, 1))
        abs_n += np.abs(naive - truth).sum(axis=(0, 1))
        sq_n += np.square(naive - truth).sum(axis=(0, 1))
        n += truth.shape[0] * truth.shape[1]

    print(f"\ncheckpoint: {ckpt}")
    print(f"{'variable':<12}{'MSE(model-space)':>18}{'MAE kW (F)':>12}{'MAE kW (naive)':>16}"
          f"{'RMSE kW (F)':>13}{'RMSE kW (naive)':>17}{'F beats naive':>15}")
    for d, name in enumerate(TARGET_NAMES):
        mae_f, mae_n = abs_f[d] / n, abs_n[d] / n
        rmse_f, rmse_n = np.sqrt(sq_f[d] / n), np.sqrt(sq_n[d] / n)
        ms = mse_model_space[d] / max(n_model_space[d], 1)
        verdict = "YES" if mae_f < mae_n else "NO  <-- investigate"
        print(f"{name:<12}{ms:>18.5f}{mae_f:>12.2f}{mae_n:>16.2f}"
              f"{rmse_f:>13.2f}{rmse_n:>17.2f}{verdict:>15}")
    print("\nnote: the printed pretrain val loss is the average of the model-space "
          "column (z-score / log1p units), NOT kW.")


if __name__ == "__main__":
    main()
