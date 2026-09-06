# NsDiff-TimeXer-Exog V3 modifications

This document records the code changes that implement the
`NsDiff-TimeXer-Exog-VarianceV2` improvement plan (V3):

```
mu_h      = SeasonalBase_h + Head_mu( Decoder( Q_h , [F_global ; F_local] ) )
sigma_h^2 = Softplus( Head_sigma( CrossAttn(Q_h, F_hist), stopgrad(mu_h) ) ) + eps
R_0       = (Y - stopgrad(mu)) / stopgrad(sigma)
R^(s)     ~ NsDiff diffusion( R_0 | history, mu )
Y^(s)     = mu + sigma * R^(s)
```

The NsDiff denoiser (`src/models/NsDiff.py`, `src/layer/denoise.py`), the
uncertainty-aware noise schedule and the reverse sampler are untouched; only
the estimators, the losses, the diffusion *endpoint definition* and the training
schedule change.

## 1. Files

| File | Change |
|---|---|
| `src/layer/timexer_exog_backbone.py` | new `SeasonalBaseline`, `LocalDilatedTCN`, `FutureHorizonQueryEmbedding`, `HorizonDecoderLayer`, `DaylightGate`, `TimeXerExogenousMeanV3`, `ResidualConditionalVarianceV3` |
| `src/experiments/NsDiff_energy_exog.py` | `estimator_variant=v3`; slope/curvature mean loss; `detach_mean_for_diffusion`; residual-NLL variance loss; `diffusion_space=residual`; staged training with `best_mean_mae.pt` / `best_prob_crps.pt`; Phase-0 diagnostics (`diagnostics.json`); Figure 1 overlays the direct mean $f_\phi$ |
| `src/datasets/energy4_exog.py` | `TargetTransform` with optional `pv_transform=log1p`; a linear scaler is kept so every reported standardized metric is on the same scale regardless of the PV transform; `max_windows_per_split` for smoke tests |
| `scripts/Energy4Exog/v3_common.sh` | argument sets = ablation columns |
| `scripts/Energy4Exog/run_v3_full.sh` | full V3 (V7) run |
| `scripts/Energy4Exog/run_v3_ablation.sh` | V0 … V7 ladder + `compare_v3_ablation.py` table |
| `scripts/Energy4Exog/diagnose_phase0.sh` | Phase 0 diagnosis of the current model |

## 2. Mean estimator V3 (`TimeXerExogenousMeanV3`)

| Plan item | Implementation |
|---|---|
| 24 h / 168 h seasonal baseline | `SeasonalBaseline`: `alpha_h * X_{t-24+h} + (1-alpha_h) * X_{t-168+h}`; `v3_seasonal_gate=fixed` uses `alpha = 0.7`, `learned` predicts `alpha_h` per target from the future weather / calendar (initialised at 0.7). The network only predicts the residual `Delta mu = Y - mu_seasonal`; the residual head is initialised small so training starts at the baseline. |
| Horizon-aligned future exogenous queries | `FutureHorizonQueryEmbedding`: `e_h = MLP([W_h, T_h]) + E_h`, 24 tokens of size `D`; `HorizonDecoderLayer` (self-attention over the 24 horizons -> cross-attention to the history memory -> FFN) x `v3_dec_layers`; horizon-specific head `Linear(D, 4)` instead of `FlattenHead`. |
| TimeXer global branch kept | exactly the previous patch-24 endogenous/variate-token/DSAttention encoder with the tau/delta projectors; its `4 x (7+1)` tokens form the global memory. |
| Local dilated-TCN branch | `LocalDilatedTCN`: causal `Conv1d -> GELU -> Dropout -> Conv1d -> residual` blocks with dilations 1,2,4,8,16, kernel 3, hidden 128, receptive field 125 h; produces 168 point-wise memory tokens. Global + local memories are concatenated (first-stage fusion of the plan). |
| PV daylight gate (optional) | `DaylightGate`: `m_h = sigmoid(a (GHI_h - b))` on the scaled future GHI; `mu_PV = m mu + (1-m) PV_floor`, `sigma_PV = m sigma_day + (1-m) sigma_night`. Off by default (`pv_daylight_gate=False`); `log1p` is the default PV constraint. |

Ablation switches: `v3_use_seasonal`, `v3_use_horizon_query`, `v3_use_local_tcn`.
With `v3_use_horizon_query=False` the module falls back to the original
`FlattenHead`, so V1 (= V0 + seasonal skip) is reproduced exactly.

## 3. Mean objective

```
L_mu = MSE(mu, Y) + lambda_slope * SmoothL1(d mu, d Y) + lambda_curve * SmoothL1(d^2 mu, d^2 Y)
```

`lambda_slope=0.3`, `lambda_curve=0.05` in the scripts (0 disables a term).

## 4. Decoupling from the diffusion loss

* `detach_mean_for_diffusion=True` (default for `v3`): `mu.detach()` is used
  for `q_sample`, for the denoiser input and for the endpoint, so
  `L_diff` no longer updates `f_phi`.
* `detach_variance_for_diffusion` defaults to `True` whenever the variance is
  trained with the residual NLL, so `L_diff` no longer updates `g_psi` either.

## 5. Variance V3 (`ResidualConditionalVarianceV3`, `variance_loss=residual_nll`)

```
r_h      = Y_h - stopgrad(mu_h)
sigma^2  = Softplus(z_h) + 1e-5
L_var    = 1/2 [ log sigma_h^2 + r_h^2 / sigma_h^2 ]
```

Query per (target, horizon): future weather + calendar, horizon embedding,
target embedding, `stopgrad(mu_h)` and (optionally) the detached decoder
feature `Z_h` of the mean model. Memory per target: six 12 h patches of the
log rolling variance plus four 12 h patches of the last 48 h of values.
The earlier estimators stay available (`variance_estimator=original |
timexer_v1 | residual_v2`) with the rolling-variance target
(`variance_loss=rolling`) for the ablation rungs V1-V4. When
`residual_v2` is used, `diagnostics.json` reports the statistics of
`delta_log_var` (mean/std/min/max and the fraction at +-0.95 bound, plan
section 13).

## 6. Standardized residual diffusion (`diffusion_space=residual`)

The diffusion model receives `R_0 = (Y - stopgrad(mu)) / stopgrad(sigma)`.
Within NsDiff this is a prior `R_T ~ N(0, I)` (`y_T_mean = 0`,
`g = 1`, `Sigma_Y0 = 1`), so the forward noise `(bar beta_t - tilde beta_t) g
+ tilde beta_t Sigma_Y0` reduces to `bar beta_t` and NsDiff's posterior
reduces to the standard DDPM posterior.  The denoiser is still conditioned on
the history, the time marks and `mu` (its `y_0_hat` input), so it can model
non-Gaussian, centre-dependent residuals (e.g. PV at night). Samples are
reconstructed as `Y^(s) = mu + sigma * R^(s)`; validation-only post-hoc
calibration is then applied exactly as before.

## 7. Training schedule (`training_schedule=staged`)

1. **Mean pre-training** (`mean_epochs`, `mean_patience`): only `f_phi`, loss
   `L_mu`, checkpoint by validation **MAE** (`best_mean_mae.pt`). No sampling
   is needed, so this stage is cheap.
2. **Variance + diffusion**: `f_phi` frozen, `L_var + L_diff`, checkpoint by
   validation **CRPS** (`best_prob_crps.pt`, plus the usual `model.pth`,
   `cond_pred_model.pth`, `cond_pred_model_g.pth`).
3. **Optional joint fine-tuning** (`joint_epochs>0`): the mean gets
   `lr * joint_mean_lr_scale`, the diffusion loss stays detached from the mean.

## 8. PV physical support

`pv_transform=log1p`: the loader forecasts `z = log(1 + PV)`; every sample is
mapped back with `expm1`, so `PV >= 0` holds without inference clamping.
All standardized metrics (CRPS, MAE, ...) are computed on the linear
standardized scale of the raw targets, so they stay comparable with the
`pv_transform=none` runs.

## 9. Phase-0 diagnostics (`diagnostics.json`, every run)

* `direct_mean_mae/rmse` vs `mc_mean_mae/rmse` (+ per variable, original units)
  and the resulting verdict (case A / case B of plan section 10);
* `slope_mae`, peak magnitude / timing / ramp errors for the direct and the
  Monte-Carlo mean;
* `picp_50/80/95`, `mpiw_50/80/95` (+ per variable);
* PV: negative-sample rate (raw and calibrated ensemble), nighttime MAE and
  95 % interval width in original units, daytime peak timing error;
* Figure 1 overlays `Direct mean f_phi` on the Monte-Carlo mean and the PIs.

## 10. Ablation ladder (`run_v3_ablation.sh`)

| Variant | Seasonal | Horizon Exog | Local TCN | Slope Loss | Residual Var | Residual Diffusion | PV Constraint |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| V0 current model | × | × | × | × | × | × | × |
| V1 | ✓ | × | × | × | × | × | × |
| V2 | ✓ | ✓ | × | × | × | × | × |
| V3 | ✓ | ✓ | × | ✓ | × | × | × |
| V4 | ✓ | ✓ | ✓ | ✓ | × | × | × |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × |
| V6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| V7 full | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (log1p) |

V1-V7 use `detach_mean_for_diffusion=True` and the staged schedule
(plan section 22, item A); V0 keeps the original joint training so that it is
exactly the current model.

## 11. Commands

```bash
# Phase 0: diagnose the current model
bash scripts/Energy4Exog/diagnose_phase0.sh
# full V3
PHYSICAL_GPU=0 bash scripts/Energy4Exog/run_v3_full.sh
# ablation ladder, three seeds
VARIANTS="V0 V1 V2 V3 V4 V5 V6 V7" SEEDS="1 2 3" bash scripts/Energy4Exog/run_v3_ablation.sh
# CPU-only machine
DEVICE=cpu NUM_WORKER=0 bash scripts/Energy4Exog/run_v3_full.sh
```
