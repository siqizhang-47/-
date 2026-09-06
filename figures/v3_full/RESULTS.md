# ExoMV-D V3 full model: results (Energy4Exog 2018-2019, seed 1, CPU run)

Configuration (`scripts/Energy4Exog/run_v3_full.sh`, ablation rung V7):
seasonal baseline (fixed 0.7/0.3) + TimeXer global branch (patch 24, d=256)
+ local dilated TCN (dilations 1-16, hidden 128) + 24 horizon-aligned future
exogenous queries + slope/curvature loss (0.3 / 0.05) + residual-NLL variance
+ standardized residual diffusion + log1p PV.  Staged training: mean stage
stopped after 20 epochs (best validation MAE 0.138, epoch 12), variance +
diffusion stage 6 epochs (best validation CRPS at epoch 1), 100 samples,
validation-only calibration (bias + spread, grid 0.6-2.4).

Figures (this directory): `fig1_prediction_intervals.png`,
`fig2_marginal_pdfs.png`, `fig3_pearson_correlation.png`.

## Test metrics (linear standardized scale)

| CRPS | QICE | ES | VS | MAE | RMSE | VMAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1312 | 2.78 % | 2.064 | 0.0390 | 0.1778 | 0.3326 | 0.673 |

MAE in original units: Electricity 1178, PV 368, Cooling 531, Heat 0.32.

## Phase-0 diagnostics (`diagnostics.json`)

| | Direct mean $f_\phi$ | Monte-Carlo mean |
|---|---:|---:|
| MAE | 0.1732 | 0.1778 |
| RMSE | 0.3079 | 0.3326 |
| Slope MAE | 0.0918 | 0.0946 |

Verdict: case B of plan section 10, the sample mean tracks the direct mean
(the residual diffusion no longer shifts the centre); remaining error is in
the mean estimator.

| PI level | PICP | MPIW |
|---|---:|---:|
| 50 % | 0.415 | 0.227 |
| 80 % | 0.661 | 0.429 |
| 95 % | 0.805 | 0.646 |

Peak timing error (h, MC mean): Electricity 5.5, PV 1.9, Cooling 2.9, Heat 4.5.

PV: negative-sample rate 0.07 % before / 0 % after calibration (log1p domain),
nighttime MAE 5.2 (original units), nighttime 95 % width 8.4, daytime peak
timing error 1.9 h.

## Notes

* Intervals are still under-dispersed (PICP-95 = 0.81): the residual NLL
  variance was trained for only 6 epochs on CPU; the plan's Phase 3 grid
  (more epochs, `joint_epochs > 0`) is the next step.
* Metrics are computed on the linear standardized scale of the raw targets so
  they remain comparable with `pv_transform=none` runs.
