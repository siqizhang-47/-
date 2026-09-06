# Variance Estimator V2

V2 keeps the original NsDiff rolling-variance MLP as `g_base` and learns only a
bounded, horizon-aware exogenous correction:

`g_v2 = g_base * exp(max_log_correction * tanh(delta_raw))`

Key changes versus the TimeXer variance V1:

- no `denormalize -> Softplus` output path;
- zero-initialized correction, so V2 starts exactly at the original `g_psi`;
- lightweight 1-layer volatility encoder (`d_model=96`, 4 heads);
- 72-point rolling-variance history is split into six 12-hour patches;
- future weather/calendar features remain aligned to each of the 24 horizons;
- predicted mean is used as detached conditioning for heteroscedasticity;
- the NsDiff diffusion network, variance target and sqrt-variance loss are unchanged.

## Fair 10-epoch run

```bash
PHYSICAL_GPU=2 bash scripts/Energy4Exog/run_mean_and_var_v2_gpu2.sh
```

## Tuned 20-epoch run

```bash
PHYSICAL_GPU=2 bash scripts/Energy4Exog/run_mean_and_var_v2_tuned_gpu2.sh
```

## Compare the three fair variants

Run `mean_only`, `mean_and_var` (V1), and `mean_and_var_v2`, then:

```bash
python scripts/Energy4Exog/compare_variants.py
```

Outputs:

- `energy4_exog_outputs/comparison.csv`
- `energy4_exog_outputs/comparison_metrics.png`
- each variant's `seed_42/metrics.json`
- `fig1_prediction_intervals.png`
- `fig2_marginal_pdfs.png`
- `fig3_pearson_correlation.png`
