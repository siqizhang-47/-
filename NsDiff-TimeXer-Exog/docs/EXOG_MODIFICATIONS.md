# Controlled exogenous-estimator modification

## Unchanged NsDiff core files

The exogenous experiments do not modify these upstream NsDiff core files:

- `src/models/NsDiff.py`
- `src/layer/denoise.py`
- `src/layer/nsdiff_utils.py`
- `src/layer/mu_backbone.py`
- `src/layer/g_backbone.py`
- `configs/nsdiff.yml`

`mean_only` simply does not instantiate `mu_backbone.Model`; it instantiates the new TimeXer-style mean estimator instead. The original `g_backbone.SigmaEstimation` remains active.

`mean_and_var` instantiates the same new mean estimator plus the new TimeXer-style variance estimator.

## Added files

- `src/layer/timexer_exog_backbone.py`
- `src/datasets/energy4_exog.py`
- `src/experiments/NsDiff_energy_exog.py`
- `scripts/Energy4Exog/*`
- `dataset/Energy4Exog/energy4_exog.csv`

The experiment wrapper changes data plumbing only so the estimator(s) can receive historical/future exogenous covariates. The diffusion denoiser receives the same target-history/time-mark interface as upstream NsDiff.
