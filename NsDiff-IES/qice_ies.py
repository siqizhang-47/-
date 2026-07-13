"""
Compute QICE (Quantile Interval Coverage Error) for the trained NsDiff model on
the IES dataset. Self-contained: loads the best checkpoint, draws the diffusion
samples over the test set and computes QICE — no training, no other files edited.

QICE (n_bins quantile intervals): for every ground-truth point find which of the
model's predictive-quantile intervals it falls in. Perfect calibration puts
1/n_bins of the points in each bin, so
        QICE = mean_bin | 1/n_bins - observed_ratio_bin |
Lower is better (0 = perfectly calibrated). Same definition as src/metrics/QICE.py.

Usage (GPU 4):

    export PYTHONPATH=./
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 qice_ies.py \
        --device=cuda:4 --seed=1 \
        --run_dir=/abs/path/to/results/runs/NsDiff4/IES/w168h1s24/1-<hash> \
        --n_bins=10
"""
import src.np_compat  # noqa: F401  (restore np.Inf etc. for NumPy>=2.0, keep first)

import os
import numpy as np
import torch
from tqdm import tqdm

from src.experiments.NsDiff_IES import NsDiffIES, N_TARGETS
from src.eval_ies import TARGET_NAMES

CKPTS = ("model.pth", "cond_pred_model.pth", "cond_pred_model_g.pth")


def qice(real, gsamp, n_bins=10):
    """QICE over flattened points. real:(...,), gsamp:(..., S). Returns scalar."""
    preds = gsamp.reshape(-1, gsamp.shape[-1])          # (N, S)
    targets = real.reshape(-1)                          # (N,)
    N = preds.shape[0]
    q_levels = np.arange(n_bins + 1) * (100.0 / n_bins)
    q = np.percentile(preds, q=q_levels, axis=1)        # (n_bins+1, N)
    membership = ((targets[None, :] - q) > 0).astype(int).sum(axis=0)  # (N,) in 0..n_bins+1
    counts = np.array([(membership == v).sum() for v in range(n_bins + 2)], dtype=np.float64)
    # fold the two outlier bins into the first / last interval
    counts[1] += counts[0]
    counts[-2] += counts[-1]
    counts = counts[1:-1]                               # (n_bins,)
    ratio = counts / N
    assert abs(ratio.sum() - 1.0) < 1e-6, "coverage ratios must sum to 1"
    return float(np.mean(np.abs(1.0 / n_bins - ratio))), ratio


def _resolve_run_dir(save_dir, dataset_type, w, h, p, seed, run_dir):
    if run_dir is not None:
        return run_dir
    base = os.path.join(save_dir, "runs", "NsDiff4", dataset_type, f"w{w}h{h}s{p}")
    cands = [d for d in sorted(__import__("glob").glob(os.path.join(base, "*")))
             if all(os.path.exists(os.path.join(d, c)) for c in CKPTS)]
    if not cands:
        raise FileNotFoundError(f"No complete checkpoint under '{base}'")
    seed_match = [d for d in cands if os.path.basename(d).startswith(f"{seed}-")]
    return max(seed_match or cands,
               key=lambda d: os.path.getmtime(os.path.join(d, "model.pth")))


@torch.no_grad()
def main(device="cuda:4", seed=1, dataset_type="IES",
         windows=168, horizon=1, pred_len=24, batch_size=32, num_worker=8,
         save_dir="./results", run_dir=None, n_bins=10, max_batches=None):
    run_dir = _resolve_run_dir(save_dir, dataset_type, windows, horizon, pred_len, seed, run_dir)

    exp = NsDiffIES(dataset_type=dataset_type, device=device, batch_size=batch_size,
                    windows=windows, horizon=horizon, pred_len=pred_len,
                    num_worker=num_worker, save_dir=save_dir)
    exp._setup_run(seed)
    exp.run_save_dir = run_dir
    exp.best_checkpoint_filepath = os.path.join(run_dir, "model.pth")
    exp.best_cond_checkpoint_filepath = os.path.join(run_dir, "cond_pred_model.pth")
    exp.best_cond_g_checkpoint_filepath = os.path.join(run_dir, "cond_pred_model_g.pth")
    print(f"[qice] loading checkpoint from: {run_dir}")
    exp._load_best_model()
    exp.model.eval(); exp.cond_pred_model.eval(); exp.cond_pred_model_g.eval()

    mean = np.asarray(exp.scaler.mean, dtype=np.float64)
    std = np.asarray(exp.scaler.std, dtype=np.float64)

    reals, samps = [], []
    loader = exp.test_loader
    with tqdm(total=len(loader.dataset), desc="qice-sampling") as pbar:
        for bi, (bx, by, ox, oy, bxm, bym) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            outs, truth = exp._process_val_batch(
                bx.to(exp.device).float(), by.to(exp.device).float(),
                bxm.to(exp.device).float(), bym.to(exp.device).float())
            outs = outs.detach().cpu().numpy().astype(np.float64)      # (B,H,N,S)
            truth = truth.detach().cpu().numpy().astype(np.float64)    # (B,H,N)
            inv_o = outs * std[None, None, :, None] + mean[None, None, :, None]
            inv_t = truth * std[None, None, :] + mean[None, None, :]
            reals.append(inv_t[:, :, :N_TARGETS])
            samps.append(inv_o[:, :, :N_TARGETS, :])
            pbar.update(bx.size(0))

    real = np.concatenate(reals, 0)     # (M,H,T)
    gsamp = np.concatenate(samps, 0)    # (M,H,T,S)

    overall, ratio = qice(real, gsamp, n_bins)
    print(f"\n===== QICE (n_bins={n_bins}) =====")
    print(f"Overall QICE: {overall:.4f}")
    print("Per-target QICE:")
    per = {}
    for t in range(N_TARGETS):
        q_t, _ = qice(real[..., t], gsamp[..., t, :], n_bins)
        per[TARGET_NAMES[t]] = q_t
        print(f"  {TARGET_NAMES[t]:<12}: {q_t:.4f}")
    print("Bin coverage ratio (ideal = %.3f each): %s"
          % (1.0 / n_bins, np.array2string(ratio, precision=3)))

    out_dir = os.path.join(save_dir, "figures", dataset_type,
                          f"w{windows}h{horizon}s{pred_len}")
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "qice.json"), "w") as f:
        json.dump({"QICE_overall": overall, "per_target": per,
                   "n_bins": n_bins, "bin_ratio": ratio.tolist()}, f, indent=2)
    print(f"\nsaved: {os.path.join(out_dir, 'qice.json')}")
    return overall


if __name__ == "__main__":
    import fire
    fire.Fire(main)
