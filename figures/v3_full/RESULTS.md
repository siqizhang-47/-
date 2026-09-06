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

## Phase-0 comparison with the current model (V0, `diagnose_phase0.sh`, seed 1, CPU)

V0 = ExoMV-D (TimeXer-Exog mean d=512 + Variance V2, joint training, 10 epochs,
CRPS checkpoint).  Outputs in `../phase0_current_model/`.

| Metric (linear standardized scale) | V0 current | V7 full V3 |
|---|---:|---:|
| CRPS | **0.1061** | 0.1312 |
| QICE | **0.99 %** | 2.78 % |
| MAE (MC mean) | **0.1477** | 0.1778 |
| RMSE (MC mean) | **0.2141** | 0.3326 |
| Direct-mean MAE $f_\phi$ | **0.1518** | 0.1732 |
| Slope MAE (MC mean) | **0.0909** | 0.0946 |
| PICP 50 / 80 / 95 % | **0.50 / 0.78 / 0.90** | 0.42 / 0.66 / 0.81 |
| MPIW 50 / 80 / 95 % | 0.233 / 0.454 / 0.710 | **0.227 / 0.429 / 0.646** |
| MAE Electricity / PV / Cooling / Heat (orig. units) | **1057 / 223** / 544 / **0.27** | 1178 / 368 / **531** / 0.32 |
| PV negative-sample rate (raw ensemble) | 20.1 % | **0.07 %** |
| PV nighttime MAE (orig. units) | 96.7 | **5.2** |
| PV nighttime 95 % PI width (orig. units) | 848.5 | **8.4** |
| PV daytime peak timing error (h) | **1.11** | 1.93 |

Variance-V2 diagnostic (plan section 13) on V0: `delta_log_var` is saturated at
the lower bound (mean = -0.600, fraction below -0.95 x bound = 100 %) for all
four variables, i.e. the bounded rolling-variance correction wants to shrink
the variance further but cannot, which confirms the motivation for the
residual-NLL variance.

Interpretation:

* The PV physical-support problem is solved by V3 (log1p domain + residual
  scale): night intervals collapse to a few kW and no negative samples remain.
* With the same CPU budget the V3 mean estimator generalises worse to the test
  period than the d=512 TimeXer mean (direct MAE 0.173 vs 0.152), and the PV
  MAE in linear units is the largest loss (368 vs 223), which is the expected
  trade-off of optimising errors in the log1p domain.  The variance /
  diffusion stage of V3 was also trained for only 6 epochs.
* Both models fall in case B of plan section 10 (sample mean ~ direct mean),
  so the remaining gap is a mean-estimator problem; the ablation ladder
  (`run_v3_ablation.sh`, especially V4 vs V0 and V6 vs V7) isolates whether the
  new mean architecture or the log1p PV domain costs the accuracy.

## Ablation rungs run on CPU (seed 1; full table in `../v3_ablation/ablation_table.md`)

| Variant | CRPS | QICE | MAE | RMSE | SlopeMAE | PICP95 | MPIW95 | Direct-mean MAE | PV MAE (orig.) | PV neg. rate | PV night W95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 current model | **0.1061** | **0.99 %** | **0.1477** | **0.2141** | 0.0909 | **0.900** | 0.710 | **0.1518** | 223 | 20.1 % | 849 |
| V4 = V3 mean + slope loss, Variance V2, absolute diffusion | 0.1220 | 1.63 % | 0.1648 | 0.2466 | 0.0868 | 0.861 | 0.642 | 0.1609 | 227 | 21.9 % | 292 |
| V6 = V4 + residual-NLL variance + residual diffusion | 0.1211 | 1.96 % | 0.1635 | 0.2495 | **0.0855** | 0.857 | **0.605** | 0.1609 | **208** | 23.1 % | 86 |
| V7 = V6 + log1p PV (full V3) | 0.1312 | 2.78 % | 0.1778 | 0.3326 | 0.0946 | 0.805 | 0.646 | 0.1732 | 368 | **0.07 %** | **8.4** |

What the ladder says (single seed, CPU budget, so differences below ~0.005 CRPS are not significant):

1. **Mean architecture (V0 -> V4).** The staged V3 mean (d = 256, seasonal
   baseline + horizon queries + TCN) has a slightly worse direct-mean MAE than
   the d = 512 TimeXer mean (0.161 vs 0.152) and this propagates to CRPS.
   The slope loss does what it was designed for (slope MAE 0.091 -> 0.087).
   The mean stage overfits quickly (train loss keeps falling while validation
   MAE stalls after ~12 epochs), so the next experiments should be
   d = 384/512, stronger dropout or weight decay, and `joint_epochs > 0`.
2. **Residual variance + residual diffusion (V4 -> V6).** Neutral on CRPS/MAE,
   sharper intervals (MPIW95 0.642 -> 0.605), best PV MAE (208), and PV night
   width shrinks from 292 to 86 - the residual scale is doing its job; it is
   just not yet well calibrated (PICP95 0.86), consistent with only 10
   variance/diffusion epochs.
3. **log1p PV (V6 -> V7).** Removes negative PV samples and collapses the
   night interval to 8 kW, but training in the log domain costs linear-scale
   PV accuracy (208 -> 368 MAE) and 0.01 CRPS.  Plan section 15.2's GHI
   daylight gate (`--pv_daylight_gate=True`, implemented) or a PV-only loss
   weighting in linear units is the alternative to test next.
