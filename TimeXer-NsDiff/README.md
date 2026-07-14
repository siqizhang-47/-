# TimeXer-NsDiff — TimeXer exogenous conditioning for NsDiff probabilistic forecasting

Implements the experiment plan: the four IES energy variables
(**Electricity, PV, Cooling, Heat**) are jointly forecast by NsDiff's
non-stationary location–scale diffusion, **conditioned** on future calendar +
weather through a **TimeXer** exogenous cross-attention encoder.

```
history energy [B,168,4] ─ patch + global tokens ─ self-attn ─▶ 4 target tokens G'
future calendar[B,24,5] ─ cyclic(9) ┐                                   │ (Query)
future weather [B,24,7] ─────────────┴─ 16 variate tokens Z_exo ─(Key/Value)
                       TimeXer cross-attention  ─▶ C_target [B,4,128]
                cond_global [B,512] ─▶ f_phi(μ) , g_psi(σ)  ─▶  NsDiff diffusion
```

- **TimeXer** parts reused from the official code (`reused/timexer_*`): patch +
  global-token endogenous embedding, positional embedding, full/attention layers.
- **NsDiff** parts reused verbatim (`reused/nsdiff_utils.py`, `denoise.py`,
  `nsdiff_core.py`): the location–scale forward/reverse diffusion and the
  conditional denoiser. The diffusion math is unchanged; TimeXer enters only
  through μ and σ (the plan's §20 minimal, easy-to-verify version).

## What it produces
- **one line of metrics** (test = 2022): `MAE RMSE sMAPE CRPS QICE
  PICP50 PICP80 PICP90 PICP95 MIW90 ES VS`
- **three figures**:
  - `fig1_pdf_kde.png` — predicted vs actual marginal distribution (histogram +
    KDE) per target;
  - `fig2_timeseries.png` — real vs generated time series (median + 90% band)
    over consecutive test days, per target;
  - `fig3_correlation.png` — Pearson correlation matrices among the four targets,
    **generated vs real** (4×4 heat-maps).

## Install & run (GPU 4)
```bash
cd TimeXer-NsDiff
pip install -r requirements.txt      # torch, numpy, pandas, matplotlib, scipy, tqdm, fire, openpyxl
bash run.sh                          # trains on cuda:4, then evaluates
```
The dataset `data/IES/aligned_energy_weather_summary.xlsx` (sheet `Aligned_Data`,
energy+weather already aligned, 78 888 hourly rows 2014-2022, no gaps) is read
directly — no merge step.

Manual equivalent:
```bash
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py train_eval --device=cuda:4 \
    --batch_size=64 --epochs=40 --patience=8 --n_samples=100
# evaluate only from a checkpoint:
python3 main.py evaluate --device=cuda:4 --ckpt=results/best.pt --n_samples=100
```
Progress bars are shown for training, validation and evaluation-sampling.
Outputs land in `results/` (`best.pt`, `metrics.txt/.json`, the two PNGs).

## Key settings (plan §2–5, §18)
| | value |
|---|---|
| history L / horizon H | 168 / 24 |
| targets | Electricity, PV, Cooling, Heat (jointly generated) |
| exogenous | calendar→9 cyclic dims + 7 weather = 16 tokens |
| split (by forecast year) | train 2014–2020 · val 2021 · **test 2022** |
| normalisation | **min-max to [0,1]**, separate target / weather scalers, **fit on train only** |
| d_model / heads / layers | 128 / 8 / 2 · patch 24 · diffusion steps 20 · S=100 |

## Important notes
- **Oracle Weather.** Evaluation uses the *true* future weather as the condition,
  so results are an upper bound on what a real weather forecast would give. They
  are labelled `[Oracle Weather]` in `metrics.txt` and the figure title — do not
  report them as operational forecast performance (plan §3, §24.6).
- **Normalised outputs.** Targets and weather are min-max scaled to **[0,1]**
  (fit on train), so the model, the metric table and all three figures are in the
  normalised [0,1] range. The 7 weather variables now include the real
  `Clearsky GHI` (M_exo = 9 calendar + 7 weather = 16).
- The denoiser conditions on μ and σ only (§20). To inject `cond_denoiser`
  directly into the denoiser (full §14), concatenate `cond["cond_denoiser"]`
  into `ConditionalGuidedModel` — the encoder already produces it.

## Metric definitions (plan §22)
Deterministic on the sample mean (raw scale): **MAE, RMSE, sMAPE**.
Probabilistic: **CRPS** (sample-based), **QICE** (calibration, mean
|1/bins − coverage|), **PICPq** (empirical coverage of the central q% interval,
q∈{50,80,90,95}), **MIW90** (mean 90% interval width), **ES** (multivariate
Energy Score over the 4 targets), **VS** (Variogram Score, order 0.5).

**All metrics and figures are in REAL (physical) units** — samples are
inverse-transformed before any score or plot. ES/VS use a capped window subsample
(`--es_vs_windows`); on real values they are dominated by the largest-magnitude
target (Electricity), which is the trade-off for physical-unit interpretability.

## Files
```
main.py                     train / evaluate / train_eval (fire CLI)
model.py                    TimeXerNsDiff (condition path + NsDiff diffusion)
timexer_nsdiff_adapter.py   TimeXer condition encoder + f_phi/g_psi heads (§27)
data_ies.py                 time-split dataset, separate scalers, cyclic calendar
metrics.py                  streaming metrics + Energy/Variogram score
figures.py                  pdf+kde / time-series / correlation heat-maps
reused/  timexer_* (TimeXer) , nsdiff_utils/denoise/nsdiff_core/sigma (NsDiff)
data/IES/  Total_energy.csv, Total_weather.csv
```
