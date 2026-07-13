"""
Only load the already-trained best NsDiff model and produce the figures +
metrics for the IES dataset. NO training happens here.

The run directory NsDiff uses is an MD5 hash of ALL training hyper-parameters
(epochs, patience, lr, batch_size, model size, ...), so a plot run whose params
differ even slightly points at a different (missing) folder. To avoid that, this
script AUTO-DISCOVERS the trained checkpoint under
    results/runs/NsDiff4/IES/w{windows}h{horizon}s{pred_len}/*/model.pth
and loads it directly, so you don't have to remember the exact training config.

Usage (GPU 4):

    export PYTHONPATH=./
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 plot_ies.py --device=cuda:4 --seed=1

    # if you have several trained runs and want a specific one:
    python3 plot_ies.py --device=cuda:4 --run_dir=results/runs/NsDiff4/IES/w168h1s24/1-<hash>

Outputs -> results/figures/IES/w{...}/ :
    fig3_profiles.png  fig4_correlation.png  fig5_pdf.png
    metrics.txt  metrics.json     # CRPS PSD APD Avg Std DTW MAE RMSE ACF
"""
import src.np_compat  # noqa: F401  (restore np.Inf etc. for NumPy>=2.0, keep first)

import os
import glob
from src.experiments.NsDiff_IES import NsDiffIES

CKPTS = ("model.pth", "cond_pred_model.pth", "cond_pred_model_g.pth")


def find_run_dir(save_dir, model_type, dataset_type, windows, horizon, pred_len, seed):
    """Locate a trained run folder that contains all three checkpoints."""
    base = os.path.join(save_dir, "runs", model_type, dataset_type,
                        f"w{windows}h{horizon}s{pred_len}")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"No runs found under '{base}'. Did training finish? "
            f"(check --windows/--horizon/--pred_len match training)")

    candidates = []
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if all(os.path.exists(os.path.join(d, c)) for c in CKPTS):
            candidates.append(d)
    if not candidates:
        have = "\n  ".join(sorted(glob.glob(os.path.join(base, "*"))) or ["<none>"])
        raise FileNotFoundError(
            f"No complete checkpoint (model.pth + cond_pred_model.pth + "
            f"cond_pred_model_g.pth) under '{base}'.\nSub-folders present:\n  {have}")

    # prefer a folder for the requested seed (run-id looks like '{seed}-<md5>')
    seed_match = [d for d in candidates if os.path.basename(d).startswith(f"{seed}-")]
    pick = (seed_match or candidates)
    # if several, take the most recently modified checkpoint
    pick = max(pick, key=lambda d: os.path.getmtime(os.path.join(d, "model.pth")))
    if len(candidates) > 1:
        print(f"[plot] {len(candidates)} trained runs found; using: {pick}")
    return pick


def main(device="cuda:4",
         seed=1,
         dataset_type="IES",
         windows=168,
         horizon=1,
         pred_len=24,
         batch_size=32,
         num_worker=8,
         save_dir="./results",
         run_dir=None,
         out_dir=None,
         max_batches=None):
    """Load best model and generate figures + metrics (no training)."""
    if run_dir is None:
        run_dir = find_run_dir(save_dir, "NsDiff4", dataset_type,
                               windows, horizon, pred_len, seed)

    exp = NsDiffIES(
        dataset_type=dataset_type,
        device=device,
        batch_size=batch_size,
        windows=windows,
        horizon=horizon,
        pred_len=pred_len,
        num_worker=num_worker,
        save_dir=save_dir,
    )
    metrics = exp.generate(seed=seed, out_dir=out_dir,
                           max_batches=max_batches, run_dir=run_dir)
    print("\nDone. Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    import fire
    fire.Fire(main)
