"""
NsDiff experiment for the merged IES dataset.

Adds a `generate` step on top of the standard NsDiff training loop: after the
best model is trained it runs the diffusion sampler over the test set, collects
S probabilistic samples per window, converts the four IES targets to per-unit
and produces Fig.3 / Fig.4 / Fig.5 plus the metric table
(CRPS, PSD, APD, Avg Std, DTW, MAE, RMSE, ACF).

Examples
--------
# 1) train, then automatically evaluate + plot (GPU 4)
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/NsDiff_IES.py \
    --dataset_type=IES --device="cuda:4" --batch_size=32 \
    --windows=168 --horizon=1 --pred_len=24 --epochs=30 \
    train_eval --seed=1

# 2) only (re)generate figures from an already-trained checkpoint
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 ./src/experiments/NsDiff_IES.py \
    --dataset_type=IES --device="cuda:4" \
    --windows=168 --horizon=1 --pred_len=24 \
    generate --seed=1
"""
import src.np_compat  # noqa: F401  (restore np.Inf etc. for NumPy>=2.0, must be first)

import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from src.experiments.NsDiff import NsDiffForecast
from src.eval_ies import make_all

# number of IES target channels evaluated/plotted (Electrical, Cooling, Heating, PV)
N_TARGETS = 4


@dataclass
class NsDiffIES(NsDiffForecast):

    def _pu_base(self):
        """Per-feature base value used to express results in per-unit (p.u.)."""
        data = np.asarray(self.dataset.data, dtype=np.float64)
        base = np.abs(data).max(axis=0)[:N_TARGETS]
        base[base == 0] = 1.0
        return base

    def _recover_start_hours(self, batch_y_mark):
        """Hour-of-day of the first predicted step, or None if not decodable.

        For freq='h' with timeenc=3 the date-encoding columns are
        [month, day, weekday, hour] and hour = (h-12)/24.
        """
        try:
            if batch_y_mark.shape[-1] != 4:
                return None
            h = batch_y_mark[:, 0, 3].detach().cpu().numpy() * 24.0 + 12.0
            return np.mod(np.rint(h).astype(int), 24)
        except Exception:
            return None

    @torch.no_grad()
    def generate(self, seed=1, out_dir=None, max_batches=None, run_dir=None):
        self._setup_run(seed)
        # Optionally point the checkpoint paths at an explicit run directory
        # (robust to hyper-parameter hash mismatches between train & plot).
        if run_dir is not None:
            self.run_save_dir = run_dir
            self.best_checkpoint_filepath = os.path.join(run_dir, "model.pth")
            self.best_cond_checkpoint_filepath = os.path.join(run_dir, "cond_pred_model.pth")
            self.best_cond_g_checkpoint_filepath = os.path.join(run_dir, "cond_pred_model_g.pth")
        print(f"[generate] loading checkpoint from: {self.run_save_dir}")
        self._load_best_model()
        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()

        mean = np.asarray(self.scaler.mean, dtype=np.float64)   # (N,)
        std = np.asarray(self.scaler.std, dtype=np.float64)      # (N,)
        base = self._pu_base()                                   # (T,)

        reals, gmeans, gsamps, hours = [], [], [], []
        have_hours = True

        loader = self.test_loader
        with tqdm(total=len(loader.dataset), desc="generate") as pbar:
            for bi, (batch_x, batch_y, origin_x, origin_y,
                     bx_mark, by_mark) in enumerate(loader):
                if max_batches is not None and bi >= max_batches:
                    break
                bx = batch_x.to(self.device).float()
                by = batch_y.to(self.device).float()
                bxm = bx_mark.to(self.device).float()
                bym = by_mark.to(self.device).float()

                outs, truth = self._process_val_batch(bx, by, bxm, bym)
                # outs: (B, pred_len, N, S) scaled ; truth: (B, pred_len, N) scaled
                outs = outs.detach().cpu().numpy().astype(np.float64)
                truth = truth.detach().cpu().numpy().astype(np.float64)

                # inverse-standardise on the feature axis (axis=2)
                inv_outs = outs * std[None, None, :, None] + mean[None, None, :, None]
                inv_truth = truth * std[None, None, :] + mean[None, None, :]

                # keep the 4 targets and convert to per-unit
                b = base[None, None, :]
                reals.append(inv_truth[:, :, :N_TARGETS] / b)
                gsamps.append(inv_outs[:, :, :N_TARGETS, :] / b[..., None])
                gmeans.append(inv_outs[:, :, :N_TARGETS, :].mean(-1) / b)

                sh = self._recover_start_hours(bym)
                if sh is None:
                    have_hours = False
                else:
                    hours.append(sh)

                pbar.update(batch_x.size(0))

        real = np.concatenate(reals, 0)          # (M, H, T)
        gmean = np.concatenate(gmeans, 0)        # (M, H, T)
        gsamp = np.concatenate(gsamps, 0)        # (M, H, T, S)
        start_hours = np.concatenate(hours, 0) if have_hours and hours else None

        if out_dir is None:
            out_dir = os.path.join(self.save_dir, "figures", self.dataset_type,
                                   f"w{self.windows}h{self.horizon}s{self.pred_len}")
        print(f"[generate] collected real={real.shape} gsamp={gsamp.shape}")
        return make_all(real, gmean, gsamp, start_hours, out_dir)

    def train_eval(self, seed=1):
        """Train one seed then produce figures + metrics."""
        self.run(seed=seed)
        return self.generate(seed=seed)


if __name__ == "__main__":
    import fire
    fire.Fire(NsDiffIES)
