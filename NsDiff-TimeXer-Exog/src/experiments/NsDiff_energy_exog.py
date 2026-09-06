"""NsDiff on Energy4 with TimeXer-style exogenous conditional estimators.

Estimator variants exposed through ``estimator_variant``:

1) ``mean_only``
   f_phi -> TimeXer-style exogenous estimator with tau/delta
   g_psi -> ORIGINAL NsDiff rolling-variance MLP

2) ``mean_and_var``
   f_phi -> same TimeXer-style exogenous estimator
   g_psi -> TimeXer-style exogenous variance estimator with tau/delta (V1)

3) ``mean_and_var_v2``
   f_phi -> same TimeXer-style exogenous estimator
   g_psi -> residual horizon-aware V2: original rolling-variance MLP baseline +
            lightweight exogenous log-variance correction

4) ``v3``  (NsDiff-TimeXer-Exog V3, see docs/V3_MODIFICATIONS.md)
   f_phi -> Seasonal baseline + TimeXer global branch + local dilated-TCN branch
            + horizon-aligned future exogenous queries (``TimeXerExogenousMeanV3``)
   g_psi -> selectable: ``residual_v3`` (forecast-residual conditional variance
            trained with a heteroscedastic Gaussian NLL) or the earlier
            ``residual_v2`` / ``timexer_v1`` / ``original`` estimators.
   Additional V3 training options (all switchable for ablations):
   * ``lambda_slope`` / ``lambda_curve``: first/second-difference SmoothL1 terms
     added to the mean MSE;
   * ``detach_mean_for_diffusion``: the diffusion loss no longer back-propagates
     into f_phi;
   * ``diffusion_space="residual"``: the diffusion model learns the standardized
     residual R = (Y - stopgrad(mu)) / sigma and samples are reconstructed as
     Y = mu + sigma * R;
   * ``training_schedule="staged"``: (1) mean pre-training with validation-MAE
     checkpointing, (2) frozen mean + variance/diffusion training with
     validation-CRPS checkpointing, (3) optional joint fine-tuning;
   * ``pv_transform="log1p"`` (dataset) and ``pv_daylight_gate`` for the PV
     physical support constraint.

Everything else is inherited from / identical to NsDiff:
* diffusion network ``src.models.NsDiff.NsDiff``;
* uncertainty-aware forward noise schedule;
* reverse sampler.

External conditions used ONLY by f_phi/g_psi:
* 168 h history of all four target variables;
* 168 h history + 24 h known future values for four weather variables;
* known calendar/time features for history + future.
The NsDiff denoising network itself still receives only the four target histories
and their original time marks (plus, as before, f_phi and g_psi).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets.energy4_exog import (
    TARGET_NAMES,
    WEATHER_NAMES,
    TIME_FEATURE_NAMES,
    make_energy4_exog_loaders,
)
from src.experiments.NsDiff_energy import NsDiffEnergyForecast, _empirical_crps_sum
from src.layer.timexer_exog_backbone import (
    TimeXerExogenousMean,
    TimeXerExogenousVariance,
    ResidualExogenousVarianceV2,
    TimeXerExogenousMeanV3,
    ResidualConditionalVarianceV3,
)
from src.layer.nsdiff_utils import q_sample, p_sample_loop, cal_sigma_tilde, cal_forward_noise
from src.utils.sigma import wv_sigma_trailing
from torch_timeseries.utils.reproduce import reproducible

EPS = 1e-7
PI_LEVELS = {"50": (0.25, 0.75), "80": (0.10, 0.90), "95": (0.025, 0.975)}


def _parse_bool_auto(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"auto", "", "none"}:
        return default
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean option {value!r}")


@dataclass
class NsDiffEnergyExogForecast(NsDiffEnergyForecast):
    # Controlled ablation switch.
    estimator_variant: str = "mean_only"  # mean_only | mean_and_var | mean_and_var_v2 | v3

    # TimeXer-style estimator hyperparameters (mean_only / mean_and_var / mean_and_var_v2).
    timexer_patch_len: int = 24
    timexer_d_model: int = 512
    timexer_n_heads: int = 8
    timexer_e_layers: int = 2
    timexer_d_ff: int = 1024

    # V2 variance hyperparameters are intentionally decoupled from the mean backbone.
    var_v2_patch_len: int = 12
    var_v2_d_model: int = 96
    var_v2_n_heads: int = 4
    var_v2_d_ff: int = 192
    var_v2_base_hidden: int = 512
    var_v2_max_log_correction: float = 1.0
    var_v2_weather_regime_len: int = 24

    # ------------------------------------------------------------------ V3 ---
    # Mean V3 architecture (used when estimator_variant == "v3").
    v3_use_seasonal: bool = True
    v3_use_horizon_query: bool = True
    v3_use_local_tcn: bool = True
    v3_seasonal_gate: str = "fixed"          # fixed | learned
    v3_seasonal_daily_weight: float = 0.7
    v3_d_model: int = 256
    v3_n_heads: int = 8
    v3_e_layers: int = 2
    v3_d_ff: int = 512
    v3_dec_layers: int = 2
    v3_tcn_hidden: int = 128
    v3_tcn_kernel: int = 3
    v3_tcn_dilations: str = "1,2,4,8,16"

    # Variance estimator / loss selection.
    #   variance_estimator: auto | original | timexer_v1 | residual_v2 | residual_v3
    #   variance_loss     : auto | rolling | residual_nll
    variance_estimator: str = "auto"
    variance_loss: str = "auto"
    var_v3_d_model: int = 128
    var_v3_n_heads: int = 4
    var_v3_d_ff: int = 256
    var_v3_patch_len: int = 12
    var_v3_recent_len: int = 48
    var_v3_use_mean_features: bool = True
    var_v3_min_var: float = 1e-5

    # Mean objective: L = MSE + lambda_slope * SmoothL1(d mu, d y) + lambda_curve * SmoothL1(d2 mu, d2 y)
    lambda_slope: float = 0.0
    lambda_curve: float = 0.0

    # Diffusion coupling.
    detach_mean_for_diffusion: str = "auto"   # auto (= True for v3, False otherwise) | true | false
    detach_variance_for_diffusion: str = "auto"  # auto (= True for residual_nll) | true | false
    diffusion_space: str = "absolute"         # absolute | residual

    # PV physical constraint.
    pv_transform: str = "none"                # none | log1p (handled in the dataset loader)
    pv_daylight_gate: bool = False

    # Training schedule.
    training_schedule: str = "joint"          # joint | staged
    mean_epochs: int = 60
    mean_patience: int = 10
    joint_epochs: int = 0
    joint_mean_lr_scale: float = 0.1

    # Inference.
    sample_minibatch: int = 0                 # 0 -> configs/nsdiff.yml testing.minisample
    # Debug / smoke tests only: evenly spaced subset of every split (0 = full data).
    max_windows_per_split: int = 0
    # Evaluate-only: path to a best_prob_crps.pt bundle (skips training, re-runs
    # validation-only calibration + final test + figures).
    eval_checkpoint: str = ""

    # Validation-only post-hoc calibration. This does NOT use test labels.
    # It corrects systematic mean bias and over/under-dispersed ensemble spread.
    enable_posthoc_calibration: bool = True
    calibration_max_windows: int = 512
    calibration_bias_strength: float = 1.0
    calibration_spread_min: float = 0.45
    calibration_spread_max: float = 1.15
    calibration_spread_steps: int = 15
    enforce_nonnegative_outputs: bool = True

    # Figure 1 selection only; metrics always use the entire test set.
    plot_example_policy: str = "best"  # first | best
    model_display_name: str = "NsDiff"

    # Default output root; run scripts use variant-specific subdirectories.
    analysis_dir: str = "./energy4_exog_outputs/mean_only"

    # ------------------------------------------------------------------ setup
    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        del fast_test, fast_val
        (
            self.dataset,
            self.scaler,
            self.weather_scaler,
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = make_energy4_exog_loaders(
            root=self.data_path,
            window=self.windows,
            pred_len=self.pred_len,
            horizon=self.horizon,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            num_workers=self.num_worker,
            shuffle_train=shuffle,
            pv_transform=self.pv_transform,
            max_windows_per_split=self.max_windows_per_split,
        )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)
        self.pv_index = TARGET_NAMES.index("PV")
        self.ghi_index = WEATHER_NAMES.index("GHI")
        self.pv_floor_scaled = float(self.scaler.transform(np.zeros((1, self.dataset.num_features)))[0, self.pv_index])
        self.ghi_night_scaled = float(self.weather_scaler.transform(np.zeros((1, self.dataset.num_weather)))[0, self.ghi_index])

    def _resolve_options(self):
        v = self.estimator_variant
        if v not in {"mean_only", "mean_and_var", "mean_and_var_v2", "v3"}:
            raise ValueError("estimator_variant must be one of mean_only | mean_and_var | mean_and_var_v2 | v3")
        default_var = {
            "mean_only": "original",
            "mean_and_var": "timexer_v1",
            "mean_and_var_v2": "residual_v2",
            "v3": "residual_v3",
        }[v]
        self._variance_estimator = default_var if self.variance_estimator == "auto" else self.variance_estimator
        if self._variance_estimator not in {"original", "timexer_v1", "residual_v2", "residual_v3"}:
            raise ValueError(f"Unknown variance_estimator {self.variance_estimator!r}")
        if self.variance_loss == "auto":
            self._variance_loss = "residual_nll" if self._variance_estimator == "residual_v3" else "rolling"
        else:
            self._variance_loss = self.variance_loss
        if self._variance_loss not in {"rolling", "residual_nll"}:
            raise ValueError("variance_loss must be auto | rolling | residual_nll")
        if self.diffusion_space not in {"absolute", "residual"}:
            raise ValueError("diffusion_space must be absolute | residual")
        if self.diffusion_space == "residual" and self._variance_loss != "residual_nll":
            raise ValueError("diffusion_space='residual' requires variance_loss='residual_nll' (sigma must be a residual scale)")
        self._detach_mean = _parse_bool_auto(self.detach_mean_for_diffusion, default=(v == "v3"))
        self._detach_var = _parse_bool_auto(
            self.detach_variance_for_diffusion, default=(self._variance_loss == "residual_nll")
        )
        if self.training_schedule not in {"joint", "staged"}:
            raise ValueError("training_schedule must be joint | staged")
        self._train_stage = "joint"
        self._mean_frozen = False

    def _init_model(self):
        self._resolve_options()
        if self.load_pretrain:
            raise ValueError(
                "load_pretrain=True is disabled for the exogenous estimator ablation because "
                "upstream pretrained f/g checkpoints have incompatible architectures."
            )

        # Build the unmodified NsDiff diffusion model + original estimators first.
        # We then replace ONLY the estimator modules requested by the experiment.
        super()._init_model()

        n_t, n_w, n_c = self.dataset.num_features, self.dataset.num_weather, self.dataset.num_time_features
        if self.estimator_variant == "v3":
            raw = self.v3_tcn_dilations
            if isinstance(raw, (tuple, list)):
                dilations = tuple(int(d) for d in raw)
            else:
                dilations = tuple(int(d) for d in str(raw).strip("()[] ").split(",") if d.strip())
            self.cond_pred_model = TimeXerExogenousMeanV3(
                seq_len=self.windows,
                pred_len=self.pred_len,
                n_targets=n_t, n_weather=n_w, n_time=n_c,
                patch_len=self.timexer_patch_len,
                d_model=self.v3_d_model,
                n_heads=self.v3_n_heads,
                e_layers=self.v3_e_layers,
                d_ff=self.v3_d_ff,
                dropout=self.dropout,
                activation=self.activation,
                p_hidden_dims=(64, 64),
                p_hidden_layers=self.p_hidden_layers,
                use_seasonal=self.v3_use_seasonal,
                use_horizon_query=self.v3_use_horizon_query,
                use_local_tcn=self.v3_use_local_tcn,
                seasonal_gate=self.v3_seasonal_gate,
                seasonal_daily_weight=self.v3_seasonal_daily_weight,
                dec_layers=self.v3_dec_layers,
                tcn_hidden=self.v3_tcn_hidden,
                tcn_kernel=self.v3_tcn_kernel,
                tcn_dilations=dilations,
                pv_daylight_gate=self.pv_daylight_gate,
                pv_index=self.pv_index,
                ghi_index=self.ghi_index,
                ghi_night_level=self.ghi_night_scaled,
                pv_floor=self.pv_floor_scaled,
            ).float().to(self.device)
            mean_feature_dim = self.v3_d_model if (self.v3_use_horizon_query and self.var_v3_use_mean_features) else None
        else:
            self.cond_pred_model = TimeXerExogenousMean(
                seq_len=self.windows,
                pred_len=self.pred_len,
                n_targets=n_t, n_weather=n_w, n_time=n_c,
                patch_len=self.timexer_patch_len,
                d_model=self.timexer_d_model,
                n_heads=self.timexer_n_heads,
                e_layers=self.timexer_e_layers,
                d_ff=self.timexer_d_ff,
                dropout=self.dropout,
                activation=self.activation,
                p_hidden_dims=(64, 64),
                p_hidden_layers=self.p_hidden_layers,
            ).float().to(self.device)
            mean_feature_dim = None

        # "original" intentionally leaves the upstream G.SigmaEstimation untouched.
        if self._variance_estimator == "timexer_v1":
            self.cond_pred_model_g = TimeXerExogenousVariance(
                seq_len=self.windows, pred_len=self.pred_len,
                n_targets=n_t, n_weather=n_w, n_time=n_c,
                rolling_length=self.rolling_length,
                patch_len=self.timexer_patch_len,
                d_model=self.timexer_d_model, n_heads=self.timexer_n_heads,
                e_layers=self.timexer_e_layers, d_ff=self.timexer_d_ff,
                dropout=self.dropout, activation=self.activation,
                p_hidden_dims=(64, 64), p_hidden_layers=self.p_hidden_layers,
            ).float().to(self.device)
        elif self._variance_estimator == "residual_v2":
            self.cond_pred_model_g = ResidualExogenousVarianceV2(
                seq_len=self.windows, pred_len=self.pred_len,
                n_targets=n_t, n_weather=n_w, n_time=n_c,
                rolling_length=self.rolling_length,
                patch_len=self.var_v2_patch_len,
                d_model=self.var_v2_d_model, n_heads=self.var_v2_n_heads, d_ff=self.var_v2_d_ff,
                dropout=self.dropout,
                base_hidden=self.var_v2_base_hidden,
                max_log_correction=self.var_v2_max_log_correction,
                weather_regime_len=self.var_v2_weather_regime_len,
            ).float().to(self.device)
        elif self._variance_estimator == "residual_v3":
            self.cond_pred_model_g = ResidualConditionalVarianceV3(
                seq_len=self.windows, pred_len=self.pred_len,
                n_targets=n_t, n_weather=n_w, n_time=n_c,
                rolling_length=self.rolling_length,
                patch_len=self.var_v3_patch_len,
                d_model=self.var_v3_d_model, n_heads=self.var_v3_n_heads, d_ff=self.var_v3_d_ff,
                dropout=self.dropout,
                min_var=self.var_v3_min_var,
                recent_len=self.var_v3_recent_len,
                mean_feature_dim=mean_feature_dim,
                pv_daylight_gate=self.pv_daylight_gate,
                pv_index=self.pv_index,
                ghi_index=self.ghi_index,
                ghi_night_level=self.ghi_night_scaled,
            ).float().to(self.device)

        print("Estimator variant:", self.estimator_variant)
        print("Variance estimator:", self._variance_estimator, "| variance loss:", self._variance_loss)
        print("Diffusion space:", self.diffusion_space,
              "| detach mean:", self._detach_mean, "| detach variance:", self._detach_var)
        print("Mean loss: MSE + %.3f*slope + %.3f*curvature" % (self.lambda_slope, self.lambda_curve))
        print("Training schedule:", self.training_schedule, "| PV transform:", self.pv_transform,
              "| PV daylight gate:", self.pv_daylight_gate)
        print("Target histories:", TARGET_NAMES)
        print("Weather exogenous variables:", WEATHER_NAMES)
        print("Time exogenous features:", TIME_FEATURE_NAMES)
        print("Future weather is supplied for the 24-step forecast horizon.")
        print("NsDiff diffusion/denoising model is unchanged.")

    def _init_optimizer(self):
        super()._init_optimizer()

    # -------------------------------------------------------------- estimators
    def _mean_forward(self, x, wx, wy, tx, ty):
        return self.cond_pred_model(x, wx, wy, tx, ty)

    def _predict_mean(self, x, wx, wy, tx, ty):
        y0, _ = self._mean_forward(x, wx, wy, tx, ty)
        return y0

    def _predict_variance(self, x, wx, wy, tx, ty, mean_hat=None, mean_features=None):
        ve = self._variance_estimator
        if ve == "timexer_v1":
            return self.cond_pred_model_g(x, wx, wy, tx, ty) + EPS
        if ve == "residual_v2":
            return self.cond_pred_model_g(x, wx, wy, tx, ty, mean_hat=mean_hat) + EPS
        if ve == "residual_v3":
            return self.cond_pred_model_g(x, wx, wy, tx, ty, mean_hat=mean_hat, mean_features=mean_features) + EPS
        # EXACT upstream variance estimator path.
        return self.cond_pred_model_g(x) + EPS

    def _mean_loss(self, mu, y):
        loss = (mu - y).square().mean()
        if self.lambda_slope > 0:
            loss = loss + self.lambda_slope * F.smooth_l1_loss(mu[:, 1:] - mu[:, :-1], y[:, 1:] - y[:, :-1])
        if self.lambda_curve > 0:
            d2m = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
            d2y = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
            loss = loss + self.lambda_curve * F.smooth_l1_loss(d2m, d2y)
        return loss

    # ------------------------------------------------------------ scale utils
    def _standardized_lower_bounds(self):
        """Physical zero expressed in the model's standardized target scale."""
        z = np.zeros((1, self.dataset.num_features), dtype=np.float32)
        return torch.as_tensor(self.scaler.transform(z)).reshape(self.dataset.num_features).float().cpu()

    def _metric_scale_points(self, scaled: torch.Tensor) -> torch.Tensor:
        """Model scale [.., N] -> linear standardized (metric) scale, CPU."""
        if self.pv_transform == "none":
            return scaled.detach().float().cpu()
        orig = self._inverse_points(scaled)
        shape = orig.shape
        lin = self.dataset.linear_target_scaler.transform(orig.reshape(-1, shape[-1]))
        return torch.as_tensor(lin).reshape(shape).float()

    def _metric_scale_samples(self, scaled: torch.Tensor) -> torch.Tensor:
        """Model scale [B,O,N,S] -> linear standardized (metric) scale, CPU."""
        if self.pv_transform == "none":
            return scaled.detach().float().cpu()
        orig = self._inverse_samples(scaled)  # B,O,N,S original units
        b, o, n, s = orig.shape
        flat = orig.permute(0, 1, 3, 2).reshape(-1, n)
        lin = torch.as_tensor(self.dataset.linear_target_scaler.transform(flat))
        return lin.reshape(b, o, s, n).permute(0, 1, 3, 2).float()

    # -------------------------------------------------------------- calibration
    def _apply_posthoc_calibration(self, preds: torch.Tensor) -> torch.Tensor:
        p = preds.detach().float().cpu()
        if not getattr(self, "_calibration_ready", False):
            if self.enforce_nonnegative_outputs:
                lower = self._standardized_lower_bounds().view(1, 1, -1, 1)
                p = torch.maximum(p, lower)
            return p
        bias = self._calibration_bias.view(1, self.pred_len, self.dataset.num_features, 1)
        spread = self._calibration_spread.view(1, 1, self.dataset.num_features, 1)
        p = p + bias
        center = p.mean(dim=-1, keepdim=True)
        p = center + spread * (p - center)
        if self.enforce_nonnegative_outputs:
            lower = self._calibration_lower.view(1, 1, self.dataset.num_features, 1)
            p = torch.maximum(p, lower)
        return p

    @torch.no_grad()
    def _fit_posthoc_calibration(self):
        self._calibration_ready = False
        if not self.enable_posthoc_calibration:
            print("Post-hoc calibration disabled.")
            return
        self.model.eval(); self.cond_pred_model.eval(); self.cond_pred_model_g.eval()
        pred_chunks, truth_chunks = [], []
        n_collected = 0
        with tqdm(total=min(len(self.val_loader.dataset), self.calibration_max_windows), desc="fit-calibration") as bar:
            for batch in self.val_loader:
                x, y, _ox, _oy, tx, ty, wx, wy = self._to_device_batch(batch, self.device)
                preds, truths = self._process_val_batch_exog(x, y, tx, ty, wx, wy)
                take = min(preds.shape[0], self.calibration_max_windows - n_collected)
                if take <= 0:
                    break
                pred_chunks.append(preds[:take].float().cpu())
                truth_chunks.append(truths[:take].detach().float().cpu())
                n_collected += take
                bar.update(take)
                if n_collected >= self.calibration_max_windows:
                    break
        if not pred_chunks:
            print("No validation samples available for calibration; using raw ensemble.")
            return
        preds = torch.cat(pred_chunks, dim=0)
        truths = torch.cat(truth_chunks, dim=0)
        pred_mean = preds.mean(dim=-1)
        bias = (truths - pred_mean).mean(dim=0) * float(self.calibration_bias_strength)
        corrected = preds + bias.unsqueeze(0).unsqueeze(-1)
        lower = self._standardized_lower_bounds()
        grid = torch.linspace(float(self.calibration_spread_min), float(self.calibration_spread_max), int(self.calibration_spread_steps))
        best_spread = torch.ones(self.dataset.num_features, dtype=torch.float32)
        for j in range(self.dataset.num_features):
            pj = corrected[:, :, j:j + 1, :]
            yj = truths[:, :, j:j + 1]
            center = pj.mean(dim=-1, keepdim=True)
            best_score, best_tau = float("inf"), 1.0
            for tau in grid.tolist():
                cand = center + float(tau) * (pj - center)
                if self.enforce_nonnegative_outputs:
                    cand = torch.clamp(cand, min=float(lower[j]))
                score_sum, score_n = _empirical_crps_sum(cand, yj)
                score = score_sum / max(score_n, 1)
                if score < best_score:
                    best_score, best_tau = score, float(tau)
            best_spread[j] = best_tau
        self._calibration_bias = bias.cpu()
        self._calibration_spread = best_spread.cpu()
        self._calibration_lower = lower.cpu()
        self._calibration_ready = True
        print("Validation-only calibration fitted:")
        print("  spread temperatures:", {TARGET_NAMES[j]: round(float(best_spread[j]), 4) for j in range(self.dataset.num_features)})
        print("  mean |bias| by variable:", {TARGET_NAMES[j]: round(float(bias[:, j].abs().mean()), 6) for j in range(self.dataset.num_features)})

    def _update_vmae(self, state, preds: torch.Tensor, x: torch.Tensor, truths: torch.Tensor):
        """VMAE = mean | Var_ensemble(Y) - RollingVar_96([history, truth]) | (metric scale)."""
        p = preds.detach().float().cpu()
        x_cpu = x.detach().float().cpu()
        y_cpu = truths.detach().float().cpu()
        target_var = wv_sigma_trailing(torch.cat([x_cpu, y_cpu], dim=1), self.rolling_length)[:, -self.pred_len:, :] + EPS
        pred_var = p.var(dim=-1, unbiased=False)
        d = (pred_var - target_var).abs()
        state["vmae_sum"] += d.sum().item()
        state["vmae_count"] += int(d.numel())

    # ----------------------------------------------------------------- training
    def _train(self):
        self.model.train()
        self.cond_pred_model.eval() if self._mean_frozen else self.cond_pred_model.train()
        self.cond_pred_model_g.train()
        train_loss = []
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset), desc=f"train-{self._train_stage}") as progress_bar:
            for batch in self.train_loader:
                x, y, _ox, _oy, tx, ty, wx, wy = self._to_device_batch(batch, self.device)
                loss = self._process_train_batch_exog(x, y, tx, ty, wx, wy)
                loss.backward()
                train_loss.append(float(loss.item()))
                progress_bar.update(x.size(0))
                progress_bar.set_postfix(loss=float(loss.item()), epoch=self.current_epoch, refresh=True)
                self.model_optim.step()
                self.model_optim.zero_grad()
        self.model.eval(); self.cond_pred_model.eval(); self.cond_pred_model_g.eval()
        return train_loss

    def _process_train_batch_exog(self, batch_x, batch_y, batch_x_mark, batch_y_mark, weather_x, weather_y):
        # SAME ground-truth local variance construction as upstream NsDiff.
        y_sigma = wv_sigma_trailing(torch.cat([batch_x, batch_y], dim=1), self.rolling_length)[:, -self.pred_len:, :] + EPS

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,), device=self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # --- conditional mean ---
        if self._mean_frozen:
            with torch.no_grad():
                mu, aux = self._mean_forward(batch_x, weather_x, weather_y, batch_x_mark, batch_y_mark)
            loss_mean = torch.zeros((), device=self.device)
        else:
            mu, aux = self._mean_forward(batch_x, weather_x, weather_y, batch_x_mark, batch_y_mark)
            loss_mean = self._mean_loss(mu, batch_y)
        if self._train_stage == "mean":
            return loss_mean
        feats = aux.get("features")

        # --- conditional variance ---
        gx = self._predict_variance(batch_x, weather_x, weather_y, batch_x_mark, batch_y_mark, mean_hat=mu, mean_features=feats)
        if self._variance_loss == "rolling":
            loss_var = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()
        else:  # heteroscedastic Gaussian NLL of the forecast residual r = Y - stopgrad(mu)
            r = batch_y - mu.detach()
            loss_var = 0.5 * (torch.log(gx) + r.square() / gx).mean()

        # --- diffusion (upstream NsDiff logic; only the endpoint definition changes) ---
        mu_d = mu.detach() if self._detach_mean else mu
        gx_d = gx.detach() if self._detach_var else gx
        if self.diffusion_space == "absolute":
            y0, prior_mean, gx_diff, y_sigma_diff = batch_y, mu_d, gx_d, y_sigma
        else:
            sigma = torch.sqrt(gx).detach()
            y0 = (batch_y - mu_d) / sigma
            prior_mean = torch.zeros_like(batch_y)
            gx_diff = torch.ones_like(batch_y)
            y_sigma_diff = torch.ones_like(batch_y)

        e = torch.randn_like(batch_y, device=self.device)
        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx_diff, y_sigma_diff, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(
            self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
            self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
            self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx_diff, y_sigma_diff, t,
        )
        y_t_batch = q_sample(y0, prior_mean, self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt, t, noise=noise)
        output, sigma_theta = self.model(batch_x, batch_x_mark, y_t_batch, mu_d, gx_diff, t)
        sigma_theta = sigma_theta + EPS
        kl_loss = (
            (e - output).square().mean()
            + (sigma_tilde / sigma_theta).mean()
            - torch.log(sigma_tilde / sigma_theta).mean()
        )
        return kl_loss + loss_mean + loss_var

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def _process_val_batch_exog(self, batch_x, batch_y, batch_x_mark, batch_y_mark, weather_x, weather_y):
        """100-sample NsDiff inference; only f/g conditioning calls are replaced."""
        b = batch_x.shape[0]
        mu, aux = self._mean_forward(batch_x, weather_x, weather_y, batch_x_mark, batch_y_mark)
        gx = self._predict_variance(batch_x, weather_x, weather_y, batch_x_mark, batch_y_mark, mean_hat=mu, mean_features=aux.get("features"))
        self._last_direct_mean = mu.detach().float().cpu()
        self._last_variance = gx.detach().float().cpu()
        if self._variance_estimator == "residual_v2":
            self._last_delta_log_var = self.cond_pred_model_g.last_delta_log_var

        if self.diffusion_space == "absolute":
            prior_mean, gx_diff = mu, gx
        else:
            prior_mean, gx_diff = torch.zeros_like(mu), torch.ones_like(gx)

        n_samples = int(self.diffusion_config.testing.n_z_samples)
        chunk = int(self.sample_minibatch) if int(self.sample_minibatch) > 0 else max(1, int(self.diffusion_config.testing.minisample))
        sample_chunks, generated = [], 0
        O, N = self.pred_len, self.dataset.num_features
        while generated < n_samples:
            r = min(chunk, n_samples - generated)
            tile = lambda z, L, D: z.unsqueeze(1).expand(-1, r, -1, -1).reshape(b * r, L, D)
            y_seq = p_sample_loop(
                self.model, tile(batch_x, self.windows, N), tile(batch_x_mark, self.windows, batch_x_mark.shape[-1]),
                tile(mu, O, N), tile(gx_diff, O, N), tile(prior_mean, O, N),
                self.model.num_timesteps, self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                self.model.alphas_cumprod, self.model.alphas_cumprod_sum, self.model.alphas_cumprod_prev,
                self.model.alphas_cumprod_sum_prev, self.model.betas_tilde, self.model.betas_bar,
                self.model.betas_tilde_m_1, self.model.betas_bar_m_1,
            )
            y0_gen = y_seq[-1].reshape(b, r, O, N)
            if self.diffusion_space == "residual":
                y0_gen = mu.unsqueeze(1) + torch.sqrt(gx).unsqueeze(1) * y0_gen  # Y = mu + sigma * R
            sample_chunks.append(y0_gen.detach().cpu())
            generated += r
        preds = torch.cat(sample_chunks, dim=1).permute(0, 2, 3, 1).contiguous()  # [B,O,N,S]
        truths = batch_y[:, -self.pred_len:, :]
        return preds, truths

    @staticmethod
    def _to_device_batch(batch, device):
        x, y, ox, oy, tx, ty, wx, wy = batch
        return (
            x.to(device).float(), y.to(device).float(), ox, oy,
            tx.to(device).float(), ty.to(device).float(),
            wx.to(device).float(), wy.to(device).float(),
        )

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def _evaluate(self, dataloader):
        self.model.eval(); self.cond_pred_model.eval(); self.cond_pred_model_g.eval()
        state = self._new_state()
        with tqdm(total=len(dataloader.dataset), desc=f"eval-{self.estimator_variant}") as progress_bar:
            for batch in dataloader:
                x, y, _ox, _oy, tx, ty, wx, wy = self._to_device_batch(batch, self.device)
                preds, truths = self._process_val_batch_exog(x, y, tx, ty, wx, wy)
                self._update_state(state, self._metric_scale_samples(preds), self._metric_scale_points(truths), extended_metrics=False)
                progress_bar.update(x.shape[0])
        return self._finalize_state(state)

    @torch.no_grad()
    def _evaluate_direct_mean(self, dataloader, desc="eval-mean"):
        """Point-forecast metrics of f_phi (no sampling) on the metric scale."""
        self.cond_pred_model.eval()
        abs_sum = sq_sum = slope_sum = 0.0
        count = slope_count = 0
        with tqdm(total=len(dataloader.dataset), desc=desc) as bar:
            for batch in dataloader:
                x, y, _ox, _oy, tx, ty, wx, wy = self._to_device_batch(batch, self.device)
                mu, _ = self._mean_forward(x, wx, wy, tx, ty)
                mu_m, y_m = self._metric_scale_points(mu), self._metric_scale_points(y)
                err = mu_m - y_m
                abs_sum += err.abs().sum().item(); sq_sum += err.square().sum().item(); count += err.numel()
                d = (mu_m[:, 1:] - mu_m[:, :-1]) - (y_m[:, 1:] - y_m[:, :-1])
                slope_sum += d.abs().sum().item(); slope_count += d.numel()
                bar.update(x.shape[0])
        return {
            "mae": abs_sum / max(count, 1),
            "rmse": float(np.sqrt(sq_sum / max(count, 1))),
            "slope_mae": slope_sum / max(slope_count, 1),
        }

    # -------------------------------------------------------------- diagnostics
    def _new_diag(self):
        n = self.dataset.num_features
        z = lambda: np.zeros(n, dtype=np.float64)
        d = {
            "count": 0, "window_count": 0,
            "direct_abs": z(), "direct_sq": z(), "mc_abs": z(), "mc_sq": z(),
            "direct_abs_orig": z(), "mc_abs_orig": z(),
            "direct_slope_abs": z(), "mc_slope_abs": z(), "slope_count": 0,
            "direct_peak_mag": z(), "mc_peak_mag": z(), "direct_peak_time": z(), "mc_peak_time": z(),
            "direct_ramp": z(), "mc_ramp": z(),
            "pv_neg_raw": 0, "pv_neg_cal": 0, "pv_sample_count": 0,
            "pv_night_abs": 0.0, "pv_night_width95": 0.0, "pv_night_count": 0,
            "pv_day_peak_time": 0.0, "pv_day_windows": 0,
            "delta_stats": None,
        }
        for lvl in PI_LEVELS:
            d[f"picp{lvl}"] = z(); d[f"mpiw{lvl}"] = z()
        return d

    def _update_diag(self, d, preds_m, mu_m, truths_m, raw_preds_orig, cal_preds_orig, mu_orig, truth_orig, wy):
        """All *_m tensors are on the metric scale: preds [B,O,N,S], mu/truths [B,O,N]."""
        B, O, N, S = preds_m.shape
        mc_m = preds_m.mean(dim=-1)
        d["count"] += B * O
        d["window_count"] += B
        d["direct_abs"] += (mu_m - truths_m).abs().sum(dim=(0, 1)).numpy()
        d["direct_sq"] += (mu_m - truths_m).square().sum(dim=(0, 1)).numpy()
        d["mc_abs"] += (mc_m - truths_m).abs().sum(dim=(0, 1)).numpy()
        d["mc_sq"] += (mc_m - truths_m).square().sum(dim=(0, 1)).numpy()
        mc_orig = cal_preds_orig.mean(dim=-1)
        d["direct_abs_orig"] += (mu_orig - truth_orig).abs().sum(dim=(0, 1)).numpy()
        d["mc_abs_orig"] += (mc_orig - truth_orig).abs().sum(dim=(0, 1)).numpy()

        diff = lambda z: z[:, 1:] - z[:, :-1]
        dy, dmu, dmc = diff(truths_m), diff(mu_m), diff(mc_m)
        d["direct_slope_abs"] += (dmu - dy).abs().sum(dim=(0, 1)).numpy()
        d["mc_slope_abs"] += (dmc - dy).abs().sum(dim=(0, 1)).numpy()
        d["slope_count"] += B * (O - 1)
        for key, z in (("direct", mu_m), ("mc", mc_m)):
            d[f"{key}_peak_mag"] += (z.max(dim=1).values - truths_m.max(dim=1).values).abs().sum(dim=0).numpy()
            d[f"{key}_peak_time"] += (z.argmax(dim=1) - truths_m.argmax(dim=1)).abs().float().sum(dim=0).numpy()
            d[f"{key}_ramp"] += (diff(z).max(dim=1).values - dy.max(dim=1).values).abs().sum(dim=0).numpy()

        for lvl, (lo, hi) in PI_LEVELS.items():
            ql = torch.quantile(preds_m, lo, dim=-1)
            qh = torch.quantile(preds_m, hi, dim=-1)
            inside = ((truths_m >= ql) & (truths_m <= qh)).float()
            d[f"picp{lvl}"] += inside.sum(dim=(0, 1)).numpy()
            d[f"mpiw{lvl}"] += (qh - ql).sum(dim=(0, 1)).numpy()

        pv = self.pv_index
        d["pv_neg_raw"] += int((raw_preds_orig[:, :, pv, :] < 0).sum().item())
        d["pv_neg_cal"] += int((cal_preds_orig[:, :, pv, :] < 0).sum().item())
        d["pv_sample_count"] += int(raw_preds_orig[:, :, pv, :].numel())
        night = (wy[..., self.ghi_index].detach().float().cpu() <= self.ghi_night_scaled + 1e-4)  # [B,O]
        if night.any():
            err_night = (mc_orig[:, :, pv] - truth_orig[:, :, pv]).abs()[night]
            q_lo = torch.quantile(cal_preds_orig[:, :, pv, :], 0.025, dim=-1)
            q_hi = torch.quantile(cal_preds_orig[:, :, pv, :], 0.975, dim=-1)
            d["pv_night_abs"] += err_night.sum().item()
            d["pv_night_width95"] += (q_hi - q_lo)[night].sum().item()
            d["pv_night_count"] += int(night.sum().item())
        day_windows = (~night).any(dim=1)
        if day_windows.any():
            pt = (mc_orig[:, :, pv].argmax(dim=1) - truth_orig[:, :, pv].argmax(dim=1)).abs().float()[day_windows]
            d["pv_day_peak_time"] += pt.sum().item()
            d["pv_day_windows"] += int(day_windows.sum().item())

        delta = getattr(self, "_last_delta_log_var", None)
        if self._variance_estimator == "residual_v2" and delta is not None:
            bound = float(self.var_v2_max_log_correction)
            if d["delta_stats"] is None:
                d["delta_stats"] = {"sum": np.zeros(N), "sq": np.zeros(N), "min": np.full(N, np.inf), "max": np.full(N, -np.inf),
                                    "low": np.zeros(N), "high": np.zeros(N), "n": 0}
            ds = d["delta_stats"]
            dl = delta.detach().float().cpu()
            ds["sum"] += dl.sum(dim=(0, 1)).numpy(); ds["sq"] += dl.square().sum(dim=(0, 1)).numpy()
            ds["min"] = np.minimum(ds["min"], dl.amin(dim=(0, 1)).numpy()); ds["max"] = np.maximum(ds["max"], dl.amax(dim=(0, 1)).numpy())
            ds["low"] += (dl < -0.95 * bound).float().sum(dim=(0, 1)).numpy(); ds["high"] += (dl > 0.95 * bound).float().sum(dim=(0, 1)).numpy()
            ds["n"] += dl.shape[0] * dl.shape[1]

    def _finalize_diag(self, d):
        names = TARGET_NAMES
        n = max(d["count"], 1); w = max(d["window_count"], 1); sc = max(d["slope_count"], 1)
        per_var = lambda arr, den: {names[i]: float(arr[i] / den) for i in range(len(names))}
        avg = lambda arr, den: float(np.mean(arr / den))
        out = {
            "scale_note": "Metric-scale values use the linear StandardScaler of the raw targets (comparable across pv_transform settings).",
            "direct_mean_mae": avg(d["direct_abs"], n), "mc_mean_mae": avg(d["mc_abs"], n),
            "direct_mean_rmse": float(np.sqrt(np.mean(d["direct_sq"] / n))), "mc_mean_rmse": float(np.sqrt(np.mean(d["mc_sq"] / n))),
            "direct_mean_mae_by_variable": per_var(d["direct_abs"], n), "mc_mean_mae_by_variable": per_var(d["mc_abs"], n),
            "direct_mean_mae_original_unit": per_var(d["direct_abs_orig"], n), "mc_mean_mae_original_unit": per_var(d["mc_abs_orig"], n),
            "slope_mae_direct": avg(d["direct_slope_abs"], sc), "slope_mae_mc": avg(d["mc_slope_abs"], sc),
            "slope_mae_by_variable_mc": per_var(d["mc_slope_abs"], sc),
            "peak_magnitude_error_direct": per_var(d["direct_peak_mag"], w), "peak_magnitude_error_mc": per_var(d["mc_peak_mag"], w),
            "peak_timing_error_hours_direct": per_var(d["direct_peak_time"], w), "peak_timing_error_hours_mc": per_var(d["mc_peak_time"], w),
            "ramp_error_direct": per_var(d["direct_ramp"], w), "ramp_error_mc": per_var(d["mc_ramp"], w),
            "pv": {
                "negative_sample_rate_raw_ensemble": d["pv_neg_raw"] / max(d["pv_sample_count"], 1),
                "negative_sample_rate_calibrated": d["pv_neg_cal"] / max(d["pv_sample_count"], 1),
                "nighttime_mae_original_unit": d["pv_night_abs"] / max(d["pv_night_count"], 1),
                "nighttime_95pi_width_original_unit": d["pv_night_width95"] / max(d["pv_night_count"], 1),
                "nighttime_hours": int(d["pv_night_count"]),
                "daytime_peak_timing_error_hours": d["pv_day_peak_time"] / max(d["pv_day_windows"], 1),
            },
        }
        for lvl in PI_LEVELS:
            out[f"picp_{lvl}"] = avg(d[f"picp{lvl}"], n); out[f"mpiw_{lvl}"] = avg(d[f"mpiw{lvl}"], n)
            out[f"picp_{lvl}_by_variable"] = per_var(d[f"picp{lvl}"], n); out[f"mpiw_{lvl}_by_variable"] = per_var(d[f"mpiw{lvl}"], n)
        if d["delta_stats"] is not None:
            ds = d["delta_stats"]; m = max(ds["n"], 1)
            mean = ds["sum"] / m
            out["variance_v2_delta_log_var"] = {
                names[i]: {"mean": float(mean[i]), "std": float(np.sqrt(max(ds["sq"][i] / m - mean[i] ** 2, 0.0))),
                           "min": float(ds["min"][i]), "max": float(ds["max"][i]),
                           "frac_below_-0.95bound": float(ds["low"][i] / m), "frac_above_+0.95bound": float(ds["high"][i] / m)}
                for i in range(len(names))
            }
        diff_mae = out["direct_mean_mae"] - out["mc_mean_mae"]
        rel = abs(diff_mae) / max(out["mc_mean_mae"], 1e-9)
        if rel < 0.05:
            verdict = "Case B: MAE(mu,Y) ~ MAE(MC mean,Y); improve the mean estimator first."
        elif diff_mae < 0:
            verdict = "Case A: MAE(mu,Y) << MAE(MC mean,Y); the diffusion shifts the centre - prioritise residual diffusion."
        else:
            verdict = "The Monte-Carlo mean beats the direct mean; calibration/diffusion is correcting f_phi bias."
        out["phase0_verdict"] = verdict
        return out

    @torch.no_grad()
    def _final_test_and_collect(self, seed: int):
        del seed
        self.model.eval(); self.cond_pred_model.eval(); self.cond_pred_model_g.eval()
        if self.plot_example_policy not in {"first", "best"}:
            raise ValueError("plot_example_policy must be 'first' or 'best'")

        state = self._new_state()
        diag = self._new_diag()
        per_var_abs = np.zeros(self.dataset.num_features, dtype=np.float64)
        per_var_sq = np.zeros(self.dataset.num_features, dtype=np.float64)
        per_var_count = 0
        example, best_example_score = None, float("inf")
        collected_truth, collected_samples = [], []
        collected = seen_windows = 0

        with tqdm(total=len(self.test_loader.dataset), desc=f"final-test-{self.estimator_variant}") as progress_bar:
            for batch in self.test_loader:
                x, y, origin_x, origin_y, tx, ty, wx, wy = self._to_device_batch(batch, self.device)
                raw_preds, truths = self._process_val_batch_exog(x, y, tx, ty, wx, wy)
                preds = self._apply_posthoc_calibration(raw_preds)
                mu = self._last_direct_mean

                preds_m, truths_m, mu_m = self._metric_scale_samples(preds), self._metric_scale_points(truths), self._metric_scale_points(mu)
                self._update_state(state, preds_m, truths_m)
                self._update_vmae(state, preds_m, self._metric_scale_points(x), truths_m)

                preds_cpu = preds.float().cpu()
                cal_orig = self._inverse_samples(preds_cpu)
                raw_orig = self._inverse_samples(raw_preds.float().cpu())
                mu_orig = self._inverse_points(mu)
                pred_mean_orig = cal_orig.mean(dim=-1)
                truth_orig = origin_y[:, -self.pred_len:, :].float().cpu()
                err_orig = pred_mean_orig - truth_orig
                per_var_abs += err_orig.abs().sum(dim=(0, 1)).numpy()
                per_var_sq += err_orig.square().sum(dim=(0, 1)).numpy()
                per_var_count += int(err_orig.shape[0] * err_orig.shape[1])
                self._update_diag(diag, preds_m, mu_m, truths_m, raw_orig, cal_orig, mu_orig, truth_orig, wy)

                window_mae = err_orig.abs().mean(dim=(1, 2))
                if self.plot_example_policy == "first" and example is None:
                    chosen = 0
                elif self.plot_example_policy == "best":
                    chosen = int(torch.argmin(window_mae).item())
                else:
                    chosen = None
                if chosen is not None:
                    score = float(window_mae[chosen].item())
                    if example is None or score < best_example_score:
                        example = {
                            "history": origin_x[chosen, -self.windows:, :].float().cpu().numpy(),
                            "truth": truth_orig[chosen].numpy(),
                            "samples": cal_orig[chosen].numpy(),
                            "direct_mean": mu_orig[chosen].numpy(),
                            "selection_policy": ("best-MAE test window; visualization only" if self.plot_example_policy == "best" else "first test window"),
                            "window_mae_original_unit_avg": score,
                        }
                        best_example_score = score

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
                        collected_samples.append(cal_orig.index_select(0, ids).numpy())
                        collected_truth.append(truth_orig.index_select(0, ids).numpy())
                        collected += len(local_ids)
                seen_windows += preds_cpu.shape[0]
                progress_bar.update(x.shape[0])

        metrics = self._finalize_state(state)
        self._diagnostics = self._finalize_diag(diag)
        per_var = {
            TARGET_NAMES[i]: {
                "mae_original_unit": float(per_var_abs[i] / max(per_var_count, 1)),
                "mse_original_unit": float(per_var_sq[i] / max(per_var_count, 1)),
            }
            for i in range(self.dataset.num_features)
        }
        truth_pool = np.concatenate(collected_truth, axis=0) if collected_truth else None
        sample_pool = np.concatenate(collected_samples, axis=0) if collected_samples else None
        return metrics, per_var, example, truth_pool, sample_pool

    # ----------------------------------------------------------------- figures
    def _plot_prediction_intervals(self, example, path):
        """Figure 1: 24 h forecast, 50/80/95 % PIs, Monte-Carlo mean, direct mean f_phi, truth."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        truth = example["truth"]
        samples = example["samples"]
        direct = example.get("direct_mean")
        mean = samples.mean(axis=-1)
        q025, q10, q25, q75, q90, q975 = np.quantile(samples, [0.025, 0.10, 0.25, 0.75, 0.90, 0.975], axis=-1)
        fx = np.arange(1, truth.shape[0] + 1)
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.fill_between(fx, q025[:, j], q975[:, j], color="#A6CEE3", alpha=0.28, label="95% PI")
            ax.fill_between(fx, q10[:, j], q90[:, j], color="#4EA3D8", alpha=0.30, label="80% PI")
            ax.fill_between(fx, q25[:, j], q75[:, j], color="#1F78B4", alpha=0.24, label="50% PI")
            ax.plot(fx, mean[:, j], color="#1F4E79", lw=2.0, label="Predictive (MC) mean")
            if direct is not None:
                ax.plot(fx, direct[:, j], color="#2CA02C", lw=1.6, ls="--", label="Direct mean $f_\\phi$")
            ax.plot(fx, truth[:, j], color="#D62728", lw=2.0, label="Ground truth")
            ax.set_title(TARGET_NAMES[j])
            ax.set_xlabel("Forecast horizon (hour)")
            ax.set_xlim(1, truth.shape[0])
            ax.grid(alpha=0.18)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
        policy = example.get("selection_policy", "unspecified")
        fig.suptitle(f"24-hour probabilistic forecast, {self.model_display_name} (display window: {policy})", fontsize=14)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)

    # --------------------------------------------------------------- outputs
    def _save_outputs(self, seed, metrics, per_var, example, truth_pool, sample_pool):
        outdir = super()._save_outputs(seed, metrics, per_var, example, truth_pool, sample_pool)
        metrics_path = os.path.join(outdir, "metrics.json")
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.update(
            {
                "dataset": "Energy4Exog-2018-2019",
                "estimator_variant": self.estimator_variant,
                "model_display_name": self.model_display_name,
                "weather_exogenous": WEATHER_NAMES,
                "time_exogenous": TIME_FEATURE_NAMES,
                "future_weather_horizon": self.pred_len,
                "future_weather_note": (
                    "The aligned future weather values in this dataset are used as known future weather inputs. "
                    "Replace them with actual NWP forecast columns when available."
                ),
                "architecture_constraint": "Only f_phi/g_psi estimators (and the diffusion endpoint definition) are modified; the NsDiff denoiser is unchanged.",
                "metric_definitions": {
                    "ES": "trajectory-level empirical Energy Score over the 24h x 4-variable vector; up to 50 ensemble members",
                    "VS": "Variogram Score with p=0.5 across unordered variable pairs at each horizon",
                    "VMAE": "MAE between calibrated ensemble variance and the 96h rolling-variance target, linear standardized scale",
                    "scale": "standardized metrics use the linear StandardScaler of the raw training targets, independent of pv_transform",
                },
                "posthoc_calibration": {
                    "enabled": bool(self.enable_posthoc_calibration),
                    "validation_only": True,
                    "calibration_max_windows": int(self.calibration_max_windows),
                    "bias_strength": float(self.calibration_bias_strength),
                    "spread_temperature_by_variable": (
                        {TARGET_NAMES[j]: float(self._calibration_spread[j]) for j in range(self.dataset.num_features)}
                        if getattr(self, "_calibration_ready", False) else None
                    ),
                    "mean_bias_by_horizon_variable": (
                        self._calibration_bias.tolist() if getattr(self, "_calibration_ready", False) else None
                    ),
                    "enforce_nonnegative_outputs": bool(self.enforce_nonnegative_outputs),
                },
                "figure1_selection_policy": self.plot_example_policy,
                "v3_config": {
                    "mean_estimator": "TimeXerExogenousMeanV3" if self.estimator_variant == "v3" else "TimeXerExogenousMean",
                    "use_seasonal": self.v3_use_seasonal if self.estimator_variant == "v3" else False,
                    "use_horizon_query": self.v3_use_horizon_query if self.estimator_variant == "v3" else False,
                    "use_local_tcn": self.v3_use_local_tcn if self.estimator_variant == "v3" else False,
                    "seasonal_gate": self.v3_seasonal_gate,
                    "seasonal_daily_weight": self.v3_seasonal_daily_weight,
                    "variance_estimator": self._variance_estimator,
                    "variance_loss": self._variance_loss,
                    "diffusion_space": self.diffusion_space,
                    "detach_mean_for_diffusion": self._detach_mean,
                    "detach_variance_for_diffusion": self._detach_var,
                    "lambda_slope": self.lambda_slope,
                    "lambda_curve": self.lambda_curve,
                    "pv_transform": self.pv_transform,
                    "pv_daylight_gate": self.pv_daylight_gate,
                    "training_schedule": self.training_schedule,
                    "mean_epochs": self.mean_epochs,
                    "mean_patience": self.mean_patience,
                    "joint_epochs": self.joint_epochs,
                    "mean_stage_history": getattr(self, "_mean_stage_history", None),
                    "best_val_mean_mae": getattr(self, "_best_val_mean_mae", None),
                },
                "variance_v2": {
                    "enabled": self._variance_estimator == "residual_v2",
                    "max_log_correction": self.var_v2_max_log_correction,
                    "formula": "g_v2 = g_base * exp(max_log_correction * tanh(delta_raw))",
                },
                "diagnostics": getattr(self, "_diagnostics", None),
            }
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if getattr(self, "_diagnostics", None) is not None:
            with open(os.path.join(outdir, "diagnostics.json"), "w", encoding="utf-8") as f:
                json.dump(self._diagnostics, f, indent=2, ensure_ascii=False)
            print("Phase-0 diagnostics:", self._diagnostics["phase0_verdict"])
        return outdir

    # ------------------------------------------------------------------- run
    def _save_stage_checkpoints(self, outdir, tag):
        os.makedirs(outdir, exist_ok=True)
        if tag == "mean":
            torch.save(self.cond_pred_model.state_dict(), os.path.join(outdir, "best_mean_mae.pt"))
        else:
            torch.save(
                {"model": self.model.state_dict(), "cond_pred_model": self.cond_pred_model.state_dict(),
                 "cond_pred_model_g": self.cond_pred_model_g.state_dict()},
                os.path.join(outdir, "best_prob_crps.pt"),
            )

    def _set_mean_frozen(self, frozen: bool):
        self._mean_frozen = bool(frozen)
        for p in self.cond_pred_model.parameters():
            p.requires_grad_(not frozen)

    def _prob_epoch_loop(self, seed, max_epochs, outdir):
        """Diffusion/variance epochs with validation-CRPS early stopping (upstream logic)."""
        start = self.current_epoch
        while self.current_epoch < start + max_epochs:
            if self.early_stopper.early_stop:
                self._run_print(f"val CRPS did not decrease for patience={self.patience} epochs; early stopping.")
                break
            t0 = time.time()
            reproducible(seed + 1000 + self.current_epoch)
            losses = self._train()
            self._run_print(f"[{self._train_stage}] Epoch {self.current_epoch + 1} cost {time.time() - t0:.1f}s, loss {np.mean(losses):.5f}")
            val = self._val()
            self.current_epoch += 1
            self.early_stopper(val["crps"], model={"model": self.model, "cond_pred_model": self.cond_pred_model, "cond_pred_model_g": self.cond_pred_model_g})
            if self.early_stopper.counter == 0:
                self._save_stage_checkpoints(outdir, "prob")

    def evaluate(self, seed=42) -> Dict[str, float]:
        """Reload ``eval_checkpoint`` and redo calibration + final test + figures only."""
        if not self.eval_checkpoint:
            raise ValueError("evaluate requires --eval_checkpoint=<path to best_prob_crps.pt>")
        self._setup_run(seed)
        bundle = torch.load(self.eval_checkpoint, map_location=self.device)
        self.model.load_state_dict(bundle["model"])
        self.cond_pred_model.load_state_dict(bundle["cond_pred_model"])
        self.cond_pred_model_g.load_state_dict(bundle["cond_pred_model_g"])
        self._run_print = lambda *a, **k: print(*a, **k)
        reproducible(seed)
        self._fit_posthoc_calibration()
        metrics, per_var, example, truth_pool, sample_pool = self._final_test_and_collect(seed)
        print(f"best_test_results: {metrics}")
        self._save_outputs(seed, metrics, per_var, example, truth_pool, sample_pool)
        return metrics

    def run(self, seed=42) -> Dict[str, float]:
        if self.training_schedule == "joint":
            return super().run(seed)

        # ----------------------------- staged training (plan sections 8 & 9)
        self._setup_run(seed)
        self._check_run_exist(seed)
        self._run_print(f"run : {self.current_run} in seed: {seed} (staged schedule)")
        outdir = os.path.join(self.analysis_dir, f"seed_{seed}")
        os.makedirs(outdir, exist_ok=True)
        for name, m in (("model", self.model), ("cond_pred_model", self.cond_pred_model), ("cond_pred_model_g", self.cond_pred_model_g)):
            self._run_print(f"{name} parameters: {sum(p.numel() for p in m.parameters())}")

        # Stage 1: mean estimator only, checkpoint by validation MAE (not CRPS).
        self._train_stage = "mean"
        self._set_mean_frozen(False)
        self.model_optim = torch.optim.Adam(self.cond_pred_model.parameters(), lr=self.lr)
        best_mae, bad, history = float("inf"), 0, []
        for epoch in range(self.mean_epochs):
            t0 = time.time()
            reproducible(seed + epoch)
            self.current_epoch = epoch
            losses = self._train()
            val = self._evaluate_direct_mean(self.val_loader, desc="val-mean")
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **{f"val_{k}": float(v) for k, v in val.items()}})
            self._run_print(f"[mean] Epoch {epoch + 1} cost {time.time() - t0:.1f}s loss {np.mean(losses):.5f} | val MAE {val['mae']:.5f} RMSE {val['rmse']:.5f} SlopeMAE {val['slope_mae']:.5f}")
            if val["mae"] < best_mae - 1e-7:
                best_mae, bad = val["mae"], 0
                self._save_stage_checkpoints(outdir, "mean")
                self._run_print(f"[mean] new best val MAE {best_mae:.5f}; saved best_mean_mae.pt")
            else:
                bad += 1
                if bad >= self.mean_patience:
                    self._run_print(f"[mean] val MAE did not improve for {self.mean_patience} epochs; stopping stage 1.")
                    break
        self._mean_stage_history = history
        self._best_val_mean_mae = best_mae
        self.cond_pred_model.load_state_dict(torch.load(os.path.join(outdir, "best_mean_mae.pt"), map_location=self.device))

        # Stage 2: freeze the mean, train variance + diffusion, checkpoint by validation CRPS.
        self._train_stage = "prob"
        self._set_mean_frozen(True)
        self.model_optim = torch.optim.Adam(
            [{"params": self.model.parameters()}, {"params": self.cond_pred_model_g.parameters()}], lr=self.lr
        )
        self.current_epoch = 0
        self.early_stopper.reset()
        self._prob_epoch_loop(seed, self.epochs, outdir)

        # Stage 3 (optional): joint fine-tuning, small mean LR, diffusion loss still detached from the mean.
        if self.joint_epochs > 0:
            self._load_best_model()
            self._train_stage = "joint"
            self._set_mean_frozen(False)
            self.model_optim = torch.optim.Adam(
                [{"params": self.model.parameters(), "lr": self.lr},
                 {"params": self.cond_pred_model_g.parameters(), "lr": self.lr},
                 {"params": self.cond_pred_model.parameters(), "lr": self.lr * self.joint_mean_lr_scale}], lr=self.lr
            )
            self.early_stopper.counter = 0
            self.early_stopper.early_stop = False
            self._prob_epoch_loop(seed, self.joint_epochs, outdir)

        self._load_best_model()
        self._set_mean_frozen(False)
        self._fit_posthoc_calibration()
        metrics, per_var, example, truth_pool, sample_pool = self._final_test_and_collect(seed)
        self._run_print(f"best_test_results: {metrics}")
        self._save_outputs(seed, metrics, per_var, example, truth_pool, sample_pool)
        return metrics


if __name__ == "__main__":
    import fire

    fire.Fire(NsDiffEnergyExogForecast)
