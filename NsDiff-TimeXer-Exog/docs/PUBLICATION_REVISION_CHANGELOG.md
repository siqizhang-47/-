# Publication revision changelog

Compared with `NsDiff-TimeXer-Exog-VarianceV2.zip`:

1. Added final-test metrics: Energy Score (ES), Variogram Score (VS), and Variance MAE (VMAE).
2. Added validation-only post-hoc mean/spread calibration and nonnegative physical projection.
3. Figure 1 now contains only the 24-hour forecast region.
4. Figure 1 can select the lowest-MAE test window for visualization; the figure labels this choice and it never affects metrics.
5. Figure 2 uses calibrated generated samples and a shared absolute KDE bandwidth.
6. Added `{1,2,3}` three-seed runner and automatic mean ± std aggregation.
7. Added publication-ready CSV/Markdown/LaTeX/PNG table outputs.
8. Summary figures are copied from the lowest-CRPS seed, with the selected seed recorded in text.
