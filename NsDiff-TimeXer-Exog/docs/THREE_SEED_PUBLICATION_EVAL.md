# Three-seed publication evaluation (ExoMV-D)

This revision adds a reproducible `{1,2,3}` seed protocol and the requested table columns:
`CRPS`, `QICE`, `ES`, `VS`, `MAE`, and `VMAE` (all lower is better).

## Metric definitions

- **CRPS**: empirical ensemble CRPS on the standardized benchmark scale.
- **QICE**: 10-bin Quantile Interval Coverage Error. JSON stores the fraction; the publication table displays percentage points (`×100`).
- **ES**: empirical Energy Score treating each 24-hour × 4-variable forecast as one multivariate trajectory vector. Up to 50 ensemble members are used for the pairwise term for efficiency.
- **VS**: Variogram Score (`p=0.5`) over unordered variable pairs at each forecast horizon, averaged across horizons/windows.
- **MAE**: MAE of the ensemble mean on the standardized scale.
- **VMAE**: variance MAE, defined as `|Var_ensemble(Y) - RollingVar_96([history, truth])|`, using the same rolling-variance target that supervises `g_psi`.

## Validation-only post-hoc calibration

After the best training checkpoint is restored, a calibration pass uses **validation data only**:

1. horizon × variable additive mean-bias correction;
2. one ensemble spread temperature per variable, selected by validation CRPS;
3. physical non-negativity projection in original target support.

The test labels are never used to fit calibration. Test metrics and figures are computed from the calibrated ensemble.

## Figures

- Figure 1 shows only the 24-hour forecast area. The default run uses the lowest-MAE test window for visualization and labels it explicitly as `best-MAE test window; visualization only`; this does not affect any metric.
- Figure 2 uses calibrated samples and a shared absolute KDE bandwidth for the real/generated curves, so differences are not artifacts of two independently chosen KDE bandwidths.
- The summary script copies figures from the seed with lowest CRPS and records the selected seed. The experiment table is always mean ± sample standard deviation over all three seeds.
