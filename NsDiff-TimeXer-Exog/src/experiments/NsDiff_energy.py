"""NsDiff experiment tailored to the user's four-channel hourly energy dataset.

Changes relative to the upstream experiment:
1. Keeps the official NsDiff model/training loss and 100-sample inference.
2. Uses a synchronous, vectorized evaluator (avoids the upstream multiprocessing
   MetricCollection update issue and greatly accelerates CRPS evaluation).
3. Reports CRPS, QICE, MAE and MSE on the standardized benchmark scale.
4. Also reports per-variable MAE/MSE in original physical units.
5. Saves three requested figures after the best checkpoint is restored.
6. Skips test-set sampling at every training epoch; test is run once at the end.
"""

from dataclasses import dataclass
import csv
import json
import os
import time
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from src.experiments.NsDiff import NsDiffForecast
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible


FEATURE_NAMES = ["Electricity", "PV", "Cooling", "Heat"]


def _empirical_crps_sum(preds: torch.Tensor, truths: torch.Tensor):
    """Return (sum CRPS, count) for ensemble predictions.

    preds: (B, O, N, S), truths: (B, O, N)
    Uses the exact empirical CRPS identity but avoids O(S^2) pair construction.
    """
    preds = preds.float()
    truths = truths.float()
    s = preds.shape[-1]
    first = (preds - truths.unsqueeze(-1)).abs().mean(dim=-1)
    sorted_preds, _ = torch.sort(preds, dim=-1)
    weights = (2 * torch.arange(1, s + 1, dtype=preds.dtype, device=preds.device) - s - 1)
    half_pairwise = (sorted_preds * weights).sum(dim=-1) / float(s * s)
    crps = first - half_pairwise
    return crps.sum().item(), crps.numel()


def _qice_counts(preds: torch.Tensor, truths: torch.Tensor, n_bins: int = 10):
    """Count truth memberships in equal-probability ensemble quantile bins."""
    p = preds.reshape(-1, preds.shape[-1]).float()
    y = truths.reshape(-1).float()
    q = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device, dtype=p.dtype)
    quantiles = torch.quantile(p, q, dim=1)  # (n_bins+1, M)
    # Number of quantile boundaries strictly below truth: 0..n_bins+1.
    membership = (y.unsqueeze(0) > quantiles).sum(dim=0)
    # Fold below-min and above-max outliers into edge bins, matching upstream QICE.
    bin_idx = (membership - 1).clamp(min=0, max=n_bins - 1).long()
    return torch.bincount(bin_idx, minlength=n_bins).cpu().numpy(), int(y.numel())



def _energy_score_sum(preds: torch.Tensor, truths: torch.Tensor, max_samples: int = 50):
    """Trajectory-level empirical Energy Score.

    Treat each 24h x variable forecast as one multivariate trajectory vector.
    Uses up to ``max_samples`` ensemble members for the pairwise term.
    """
    preds = preds.detach().float().cpu()
    truths = truths.detach().float().cpu()
    s = min(int(preds.shape[-1]), int(max_samples))
    x = preds[..., :s].permute(0, 3, 1, 2).reshape(preds.shape[0], s, -1)
    y = truths.reshape(truths.shape[0], -1)
    first = torch.linalg.vector_norm(x - y.unsqueeze(1), ord=2, dim=-1).mean(dim=1)
    pair = torch.cdist(x, x, p=2).mean(dim=(1, 2))
    es = first - 0.5 * pair
    return es.sum().item(), int(es.numel())


def _variogram_score_sum(preds: torch.Tensor, truths: torch.Tensor, p: float = 0.5):
    """Variogram Score across the four variables at each forecast horizon.

    The score is averaged over all unordered variable pairs and all horizons.
    This keeps the metric focused on cross-variable dependence rather than
    conflating it with long-range temporal pair selection.
    """
    preds = preds.detach().float().cpu()  # B,O,N,S
    truths = truths.detach().float().cpu()  # B,O,N
    n = preds.shape[2]
    if n < 2:
        return 0.0, 0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            obs = (truths[:, :, i] - truths[:, :, j]).abs().pow(p)
            ens = (preds[:, :, i, :] - preds[:, :, j, :]).abs().pow(p).mean(dim=-1)
            v = (obs - ens).square()
            total += v.sum().item()
            count += int(v.numel())
    return total, count


@dataclass
class NsDiffEnergyForecast(NsDiffForecast):
    analysis_dir: str = "./energy4_outputs"
    plot_max_windows: int = 512
    plot_stride: int = 1
    plot_seed: int = 2026
    figure_dpi: int = 180

    def _init_metrics(self):
        # The parent class starts a 32-process pool.  We intentionally avoid it;
        # this subclass computes all requested metrics directly and synchronously.
        self.metrics = None

    @staticmethod
    def _new_state():
        return {
            "crps_sum": 0.0,
            "count": 0,
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "qice_counts": np.zeros(10, dtype=np.int64),
            "qice_count": 0,
            "es_sum": 0.0,
            "es_count": 0,
            "vs_sum": 0.0,
            "vs_count": 0,
            # VMAE is populated by the Energy4Exog subclass because it needs
            # the same rolling-variance target used to supervise g_psi.
            "vmae_sum": 0.0,
            "vmae_count": 0,
        }

    @staticmethod
    def _finalize_state(state):
        n = max(state["count"], 1)
        ratios = state["qice_counts"].astype(np.float64) / max(state["qice_count"], 1)
        qice = float(np.mean(np.abs(ratios - 0.1)))
        out = {
            "crps": float(state["crps_sum"] / n),
            "qice": qice,
            "es": float(state["es_sum"] / max(state["es_count"], 1)),
            "vs": float(state["vs_sum"] / max(state["vs_count"], 1)),
            "mae": float(state["abs_sum"] / n),
            "vmae": float(state["vmae_sum"] / max(state["vmae_count"], 1)),
            "mse": float(state["sq_sum"] / n),
        }
        return out

    def _update_state(self, state, preds, truths, extended_metrics=True):
        # Upstream sampler returns CPU predictions and GPU truths.
        preds = preds.detach().float().cpu()
        truths = truths.detach().float().cpu()
        csum, count = _empirical_crps_sum(preds, truths)
        state["crps_sum"] += csum
        state["count"] += count

        pred_mean = preds.mean(dim=-1)
        err = pred_mean - truths
        state["abs_sum"] += err.abs().sum().item()
        state["sq_sum"] += err.square().sum().item()

        counts, qn = _qice_counts(preds, truths, n_bins=10)
        state["qice_counts"] += counts
        state["qice_count"] += qn

        if extended_metrics:
            es_sum, es_count = _energy_score_sum(preds, truths, max_samples=50)
            state["es_sum"] += es_sum
            state["es_count"] += es_count
            vs_sum, vs_count = _variogram_score_sum(preds, truths, p=0.5)
            state["vs_sum"] += vs_sum
            state["vs_count"] += vs_count

    @torch.no_grad()
    def _evaluate(self, dataloader):
        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()
        state = self._new_state()

        with tqdm(total=len(dataloader.dataset), desc="probabilistic-eval") as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                self._update_state(state, preds, truths)
                progress_bar.update(batch_x.shape[0])

        return self._finalize_state(state)

    def _inverse_points(self, scaled: torch.Tensor) -> torch.Tensor:
        """Inverse-transform a tensor whose last dimension is the 4 channels."""
        shape = scaled.shape
        if shape[-1] != self.dataset.num_features:
            raise ValueError(f"Expected last dimension {self.dataset.num_features}, got {shape}")
        flat = scaled.reshape(-1, shape[-1])
        inv = self.scaler.inverse_transform(flat)
        if not torch.is_tensor(inv):
            inv = torch.as_tensor(inv)
        return inv.reshape(shape).detach().cpu()

    def _inverse_samples(self, scaled_samples: torch.Tensor) -> torch.Tensor:
        """(B,O,N,S) standardized samples -> same shape in original units."""
        b, o, n, s = scaled_samples.shape
        # Move channel to last axis before passing to a channel-wise scaler.
        flat = scaled_samples.permute(0, 1, 3, 2).reshape(-1, n)
        inv = self.scaler.inverse_transform(flat)
        if not torch.is_tensor(inv):
            inv = torch.as_tensor(inv)
        return inv.reshape(b, o, s, n).permute(0, 1, 3, 2).detach().cpu()

    @torch.no_grad()
    def _final_test_and_collect(self, seed: int):
        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()

        state = self._new_state()
        per_var_abs = np.zeros(self.dataset.num_features, dtype=np.float64)
        per_var_sq = np.zeros(self.dataset.num_features, dtype=np.float64)
        per_var_count = 0

        example = None
        collected_truth = []
        collected_samples = []
        collected = 0
        seen_windows = 0

        with tqdm(total=len(self.test_loader.dataset), desc="final-test") as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in self.test_loader:
                bx_gpu = batch_x.to(self.device).float()
                by_gpu = batch_y.to(self.device).float()
                bxmark_gpu = batch_x_date_enc.to(self.device).float()
                bymark_gpu = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(bx_gpu, by_gpu, bxmark_gpu, bymark_gpu)
                self._update_state(state, preds, truths)

                preds_cpu = preds.detach().float().cpu()
                pred_mean_orig = self._inverse_points(preds_cpu.mean(dim=-1))
                truth_orig = origin_y[:, -self.pred_len:, :].detach().float().cpu()
                err_orig = pred_mean_orig - truth_orig
                per_var_abs += err_orig.abs().sum(dim=(0, 1)).numpy()
                per_var_sq += err_orig.square().sum(dim=(0, 1)).numpy()
                per_var_count += int(err_orig.shape[0] * err_orig.shape[1])

                # Save one complete forecast-window example for Figure 1.
                if example is None:
                    sample_orig = self._inverse_samples(preds_cpu[:1])[0]  # (O,N,S)
                    example = {
                        "history": origin_x[0, -self.windows:, :].detach().float().cpu().numpy(),
                        "truth": truth_orig[0].numpy(),
                        "samples": sample_orig.numpy(),
                    }

                # Collect a bounded set of test windows for PDF/correlation figures.
                if collected < self.plot_max_windows:
                    local_ids = []
                    for local_i in range(preds_cpu.shape[0]):
                        global_i = seen_windows + local_i
                        if global_i % max(self.plot_stride, 1) == 0:
                            local_ids.append(local_i)
                            if collected + len(local_ids) >= self.plot_max_windows:
                                break
                    if local_ids:
                        ids = torch.as_tensor(local_ids, dtype=torch.long)
                        sub_samples = self._inverse_samples(preds_cpu.index_select(0, ids)).numpy()
                        sub_truth = truth_orig.index_select(0, ids).numpy()
                        collected_samples.append(sub_samples)
                        collected_truth.append(sub_truth)
                        collected += len(local_ids)

                seen_windows += preds_cpu.shape[0]
                progress_bar.update(batch_x.shape[0])

        metrics = self._finalize_state(state)
        per_var = {
            FEATURE_NAMES[i]: {
                "mae_original_unit": float(per_var_abs[i] / max(per_var_count, 1)),
                "mse_original_unit": float(per_var_sq[i] / max(per_var_count, 1)),
            }
            for i in range(self.dataset.num_features)
        }
        truth_pool = np.concatenate(collected_truth, axis=0) if collected_truth else None
        sample_pool = np.concatenate(collected_samples, axis=0) if collected_samples else None
        return metrics, per_var, example, truth_pool, sample_pool

    def _save_outputs(self, seed, metrics, per_var, example, truth_pool, sample_pool):
        outdir = os.path.join(self.analysis_dir, f"seed_{seed}")
        os.makedirs(outdir, exist_ok=True)

        payload = {
            "dataset": "Energy4",
            "features": FEATURE_NAMES,
            "window": self.windows,
            "pred_len": self.pred_len,
            "horizon": self.horizon,
            "n_samples": int(self.diffusion_config.testing.n_z_samples),
            "standardized_metrics": metrics,
            "per_variable_original_unit": per_var,
            "note": "CRPS/MAE/MSE use the StandardScaler benchmark scale; QICE is calibration error. Per-variable MAE/MSE are also reported in original units.",
        }
        with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(os.path.join(outdir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value", "scale"])
            for k in ["crps", "qice", "es", "vs", "mae", "vmae", "mse"]:
                w.writerow([k.upper(), metrics[k], "standardized"])

        with open(os.path.join(outdir, "per_variable_original_unit_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variable", "MAE", "MSE", "scale"])
            for name in FEATURE_NAMES:
                w.writerow([name, per_var[name]["mae_original_unit"], per_var[name]["mse_original_unit"], "original"])

        if example is not None:
            self._plot_prediction_intervals(example, os.path.join(outdir, "fig1_prediction_intervals.png"))
        if truth_pool is not None and sample_pool is not None:
            self._plot_pdfs(truth_pool, sample_pool, os.path.join(outdir, "fig2_marginal_pdfs.png"))
            self._plot_correlations(truth_pool, sample_pool, os.path.join(outdir, "fig3_pearson_correlation.png"))

        print(f"\nSaved Energy4 analysis to: {outdir}")
        return outdir

    def _plot_prediction_intervals(self, example, path):
        """Plot only the 24-hour forecast region (no history panel)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        truth = example["truth"]
        samples = example["samples"]  # (O,N,S), already calibrated if enabled
        mean = samples.mean(axis=-1)
        q025, q10, q25, q75, q90, q975 = np.quantile(
            samples, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975], axis=-1
        )
        fx = np.arange(1, truth.shape[0] + 1)

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.fill_between(fx, q025[:, j], q975[:, j], color="#A6CEE3", alpha=0.28, label="95% PI")
            ax.fill_between(fx, q10[:, j], q90[:, j], color="#4EA3D8", alpha=0.30, label="80% PI")
            ax.fill_between(fx, q25[:, j], q75[:, j], color="#1F78B4", alpha=0.24, label="50% PI")
            ax.plot(fx, mean[:, j], color="#1F4E79", lw=2.0, label="Predictive mean")
            ax.plot(fx, truth[:, j], color="#D62728", lw=2.0, label="Ground truth")
            ax.set_title(FEATURE_NAMES[j])
            ax.set_xlabel("Forecast horizon (hour)")
            ax.set_xlim(1, truth.shape[0])
            ax.grid(alpha=0.18)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        policy = example.get("selection_policy", "unspecified")
        fig.suptitle(f"24-hour probabilistic forecast (display window: {policy})", fontsize=14)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_pdfs(self, truth_pool, sample_pool, path):
        """Marginal PDFs with a shared absolute KDE bandwidth per variable.

        Using the same smoothing bandwidth for real/generated samples prevents
        different Scott factors from creating artificial visual discrepancies.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        rng = np.random.default_rng(self.plot_seed)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for j, ax in enumerate(axes.flat):
            real = truth_pool[:, :, j].reshape(-1)
            gen = sample_pool[:, :, j, :].reshape(-1)
            if real.size > 50000:
                real = rng.choice(real, 50000, replace=False)
            if gen.size > 50000:
                gen = rng.choice(gen, 50000, replace=False)

            # Shared absolute KDE bandwidth derived from pooled data.
            pooled = np.concatenate([real, gen])
            pooled_std = float(np.std(pooled, ddof=1)) + 1e-12
            pooled_n = max(int(pooled.size), 2)
            scott = pooled_n ** (-1.0 / 5.0)
            abs_bw = max(scott * pooled_std, 1e-8)
            rstd = float(np.std(real, ddof=1)) + 1e-12
            gstd = float(np.std(gen, ddof=1)) + 1e-12
            real_kde = gaussian_kde(real, bw_method=abs_bw / rstd)
            gen_kde = gaussian_kde(gen, bw_method=abs_bw / gstd)

            # Robust x-range: include essentially all mass but avoid a single
            # extreme Monte-Carlo draw dominating the axis.
            lo = float(min(np.quantile(real, 0.001), np.quantile(gen, 0.001)))
            hi = float(max(np.quantile(real, 0.999), np.quantile(gen, 0.999)))
            pad = 0.04 * (hi - lo + 1e-12)
            xs = np.linspace(lo - pad, hi + pad, 450)
            yr = real_kde(xs)
            yg = gen_kde(xs)
            ax.fill_between(xs, yr, alpha=0.42, color="#FF6B6B", label="Real sample")
            ax.plot(xs, yr, color="#FF4D4D", lw=1.6)
            ax.fill_between(xs, yg, alpha=0.40, color="#9E9E9E", label="Generated sample")
            ax.plot(xs, yg, color="#666666", lw=1.6)
            ax.set_title(FEATURE_NAMES[j])
            ax.set_xlabel(FEATURE_NAMES[j])
            ax.set_ylabel("PDF")
            ax.legend()
            ax.grid(alpha=0.15)
        fig.suptitle("Marginal PDFs of real and calibrated NsDiff-generated samples", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_correlations(self, truth_pool, sample_pool, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(self.plot_seed)
        real = truth_pool.reshape(-1, self.dataset.num_features)
        # Preserve cross-variable alignment for each generated ensemble member.
        gen = sample_pool.transpose(0, 1, 3, 2).reshape(-1, self.dataset.num_features)
        if real.shape[0] > 100000:
            real = real[rng.choice(real.shape[0], 100000, replace=False)]
        if gen.shape[0] > 100000:
            gen = gen[rng.choice(gen.shape[0], 100000, replace=False)]
        corr_gen = np.corrcoef(gen, rowvar=False)
        corr_real = np.corrcoef(real, rowvar=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, corr, title in [
            (axes[0], corr_gen, "Generated data"),
            (axes[1], corr_real, "Real data"),
        ]:
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
            ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=30, ha="right")
            ax.set_yticks(range(len(FEATURE_NAMES)), FEATURE_NAMES)
            ax.set_title(f"Pearson correlation matrix ({title})")
            for r in range(corr.shape[0]):
                for c in range(corr.shape[1]):
                    ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center",
                            color="white" if abs(corr[r, c]) > 0.55 else "black", fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Pearson correlation matrices of real and NsDiff-generated samples", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)

    def run(self, seed=42) -> Dict[str, float]:
        """Train with validation CRPS early stopping; test only once at the end."""
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss did not decrease for patience={self.patience} epochs; early stopping."
                )
                break

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            val_result = self._val()
            self.current_epoch += 1
            self.early_stopper(
                val_result["crps"],
                model={
                    "model": self.model,
                    "cond_pred_model": self.cond_pred_model,
                    "cond_pred_model_g": self.cond_pred_model_g,
                },
            )
            self._save_run_check_point(seed)

        self._load_best_model()
        # Optional subclass hook: fit post-hoc calibration strictly on validation data.
        if hasattr(self, "_fit_posthoc_calibration"):
            self._fit_posthoc_calibration()
        metrics, per_var, example, truth_pool, sample_pool = self._final_test_and_collect(seed)
        self._run_print(f"best_test_results: {metrics}")
        self._save_outputs(seed, metrics, per_var, example, truth_pool, sample_pool)
        return metrics


if __name__ == "__main__":
    import fire
    fire.Fire(NsDiffEnergyForecast)
