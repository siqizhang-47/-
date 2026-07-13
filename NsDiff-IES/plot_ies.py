"""
Only load the already-trained best NsDiff model and produce the figures +
metrics for the IES dataset. NO training happens here.

Usage (GPU 4, must match the hyper-parameters the model was trained with):

    export PYTHONPATH=./
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 plot_ies.py \
        --device=cuda:4 --seed=1 \
        --windows=168 --horizon=1 --pred_len=24

Outputs (results/figures/IES/w168h1s24/):
    fig3_profiles.png   fig4_correlation.png   fig5_pdf.png
    metrics.txt         metrics.json      # CRPS PSD APD Avg Std DTW MAE RMSE ACF

Notes
-----
* windows / horizon / pred_len / batch_size and the model-size flags MUST be the
  same as used at training time, otherwise the checkpoint path won't be found or
  the weights won't fit the architecture. Defaults below match scripts/NSDiff/IES.sh.
* The best checkpoint is read from
    results/runs/NsDiff4/IES/w{windows}h{horizon}s{pred_len}/<run-id>/model.pth
  (+ cond_pred_model.pth, cond_pred_model_g.pth), which `generate` loads via
  `_setup_run` -> `_load_best_model`.
* To make the PDFs / correlation matrices richer at the cost of a slower run,
  lower `testing.n_z_samples` in configs/nsdiff.yml or set fast_test=False.
"""
import src.np_compat  # noqa: F401  (restore np.Inf etc. for NumPy>=2.0, keep first)

import os
from src.experiments.NsDiff_IES import NsDiffIES


def main(device="cuda:4",
         seed=1,
         dataset_type="IES",
         windows=168,
         horizon=1,
         pred_len=24,
         batch_size=32,
         num_worker=8,
         out_dir=None,
         max_batches=None):
    """Load best model and generate figures + metrics (no training)."""
    exp = NsDiffIES(
        dataset_type=dataset_type,
        device=device,
        batch_size=batch_size,
        windows=windows,
        horizon=horizon,
        pred_len=pred_len,
        num_worker=num_worker,
    )
    metrics = exp.generate(seed=seed, out_dir=out_dir, max_batches=max_batches)
    print("\nDone. Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    import fire
    fire.Fire(main)
