# NsDiff + TimeXer-style exogenous estimators

This repository is a controlled modification of the supplied **NsDiff** code, using the supplied **TimeXer** implementation as the architectural reference for the two conditional estimators.

The purpose is to compare two variants while keeping the NsDiff diffusion model unchanged:

| Variant | conditional mean `f_phi` | conditional variance `g_psi` | NsDiff diffusion model |
|---|---|---|---|
| `mean_only` | TimeXer-style + exogenous inputs + tau/delta | **original NsDiff MLP** | unchanged |
| `mean_and_var` | TimeXer-style + exogenous inputs + tau/delta | TimeXer-style + exogenous inputs + tau/delta | unchanged |

This makes the comparison interpretable: the first experiment measures the effect of changing only the mean estimator, and the second measures the additional effect of changing the variance estimator.

## 1. Dataset and conditioning information

The included CSV is:

`dataset/Energy4Exog/energy4_exog.csv`

It contains **2018-01-01 00:00:00 to 2019-12-31 23:00:00**, 17,520 continuous hourly rows.

### Forecast targets (and only forecast outputs)

1. `Electricity`
2. `PV`
3. `Cooling`
4. `Heat`

### Weather exogenous variables

1. `Temperature`
2. `Dew Point`
3. `Humidity`
4. `GHI`

For every forecast window, the estimators receive:

- 168 h historical values of all four target variables;
- 168 h historical weather;
- 24 h future weather values;
- historical + future calendar/time features.

The four time features follow the Time-Series-Library / TimeXer hourly encoding and are scaled into `[-0.5, 0.5]`:

- HourOfDay
- DayOfWeek
- DayOfMonth
- DayOfYear

**Important:** the uploaded aligned dataset contains weather observations rather than a separate NWP forecast product. The current experiment therefore uses the aligned weather values over the future 24 h as *known future weather / perfect-weather-forecast inputs*. If you later have actual forecast columns, replace the future-weather columns in the loader; the estimator interfaces do not need to change.

## 2. Mean estimator modification

File:

`src/layer/timexer_exog_backbone.py`

Class:

`TimeXerExogenousMean`

### Endogenous branch

Each of the four target histories is split into non-overlapping patches:

- history = 168 h
- patch length = 24 h
- 7 temporal patch tokens per target
- + 1 learnable global token per target

Thus each target has 8 endogenous tokens.

### Exogenous variate tokens

Following TimeXer, each conditioning series is compressed into one variate-level token. The context token set contains:

- 4 target-history variate tokens (so each target can access the other target histories);
- 4 weather tokens constructed from history + future weather (168+24=192 h);
- 4 time-feature tokens constructed from history + future calendar features.

Total cross-attention context: **12 variate tokens**.

### Attention pathway

For each target:

1. self-attention over 7 patch tokens + global token;
2. only the global token queries the 12 exogenous/context tokens by cross-attention;
3. feed-forward network;
4. repeat for `e_layers` blocks;
5. flatten all 8 endogenous tokens and project to the next 24 target values.

This follows the central TimeXer design: **patch-wise endogenous temporal modeling + variate-wise exogenous cross-attention + global token bridge**.

## 3. tau and delta are retained

The original NsDiff mean estimator is a Non-stationary Transformer. Its original `Projector` modules for `tau` and `delta` are retained.

- `tau` remains a positive, sample-dependent scalar and rescales pre-softmax attention scores.
- `delta` remains a learned temporal de-stationary vector of length 168.

Because TimeXer attends over patches / variate tokens rather than 168 point tokens, `delta` is adapted without adding any auxiliary loss:

- endogenous self-attention: average `delta` inside each 24 h patch and append the global mean, producing 8 biases;
- exogenous cross-attention: a trainable linear projection maps the 168-dimensional temporal `delta` into 12 variate-token biases.

Both self-attention and cross-attention use DSAttention. The tau/delta parameters are optimized only through the original forecasting objective, as before.

## 4. Variance estimator modification

Only the `mean_and_var` variant replaces `g_psi`.

Class:

`TimeXerExogenousVariance`

The variance estimator preserves the upstream NsDiff variance target construction:

1. compute trailing-window variance of the historical target sequence;
2. rolling window = 96;
3. with history 168, retain 72 variance positions;
4. split them into three 24-point patches;
5. add a global token;
6. cross-attend to the same target-history + weather + time variate tokens;
7. project to 24 future variance values;
8. use `Softplus` to guarantee positivity.

The `mean_only` variant does **not** use this class: it keeps the original `src/layer/g_backbone.py::SigmaEstimation` unchanged.

## 5. Training objective is unchanged

The estimator losses remain exactly the NsDiff losses:

Mean estimator:

`loss_mean = MSE(f_phi(X, exog), Y)`

Variance estimator:

`loss_var = MSE(sqrt(g_psi(X, exog)), sqrt(sigma_Y))`

NsDiff diffusion loss is unchanged:

`loss_diff = MSE(eps, eps_theta) + mean(sigma_tilde/sigma_theta) - mean(log(sigma_tilde/sigma_theta))`

Total:

`loss = loss_diff + loss_mean + loss_var`

No new tau/delta loss, exogenous loss, or auxiliary prediction target has been added.

The diffusion denoiser still receives only:

- four historical target variables;
- original time marks;
- `Y_t`, `f_phi`, `g_psi`, diffusion step `t`.

Weather/future weather is **not** passed into the NsDiff denoising model. It is used only inside the estimator(s), exactly as requested.

## 6. Environment

The legacy NsDiff dependencies are not NumPy-2 compatible, so `requirements.txt` pins:

`numpy==1.26.4`

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If your server already has a compatible CUDA-enabled PyTorch installed, do not replace it unnecessarily.

## 7. Run experiment A: only mean estimator modified

Physical GPU 2:

```bash
bash scripts/Energy4Exog/run_mean_only_gpu2.sh
```

Equivalent key settings:

- window = 168
- pred_len = 24
- rolling variance = 96
- TimeXer patch = 24
- 100 NsDiff samples
- 20 diffusion steps
- chronological 70/10/20 split
- seed = 42

Results:

`energy4_exog_outputs/mean_only/seed_42/`

## 8. Run experiment B: mean + variance estimators modified

```bash
bash scripts/Energy4Exog/run_mean_and_var_gpu2.sh
```

Results:

`energy4_exog_outputs/mean_and_var/seed_42/`

## 9. Compare the two variants

After both runs finish:

```bash
python scripts/Energy4Exog/compare_variants.py
```

It writes:

`energy4_exog_outputs/comparison.csv`

The comparison contains CRPS, QICE, MAE and MSE and the difference:

`mean_and_var - mean_only`

For all four metrics, lower is better, so a negative delta means the additional variance-estimator modification improved that metric.

## 10. Figures and metrics

Each experiment still generates the same requested files:

- `metrics.json`
- `metrics.csv`
- `per_variable_original_unit_metrics.csv`
- `fig1_prediction_intervals.png`
- `fig2_marginal_pdfs.png`
- `fig3_pearson_correlation.png`

## 11. Main modified / added files

- `src/layer/timexer_exog_backbone.py` — the two new estimators.
- `src/datasets/energy4_exog.py` — target/weather/time window loader.
- `src/experiments/NsDiff_energy_exog.py` — controlled ablation experiment; NsDiff diffusion logic retained.
- `scripts/Energy4Exog/run_mean_only_gpu2.sh`
- `scripts/Energy4Exog/run_mean_and_var_gpu2.sh`
- `scripts/Energy4Exog/compare_variants.py`
- `dataset/Energy4Exog/energy4_exog.csv`

The original `src/models/NsDiff.py`, `src/layer/denoise.py`, `src/layer/nsdiff_utils.py` and diffusion configuration are not modified for this experiment.

## 12. V3 (2026-09)

The V3 revision (seasonal baseline + horizon-aligned exogenous queries + local
dilated TCN + slope/curvature loss + residual-NLL variance + standardized
residual diffusion + log1p PV) is documented in `docs/V3_MODIFICATIONS.md`.
Run it with `bash scripts/Energy4Exog/run_v3_full.sh`; the ablation ladder is
`bash scripts/Energy4Exog/run_v3_ablation.sh`.
