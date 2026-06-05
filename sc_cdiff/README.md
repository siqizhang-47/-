# SC-CDiff — Structure- and Correlation-aware Conditional Diffusion

Reference implementation of the scenario-forecasting method in
*SC_CDiff_TII_revised_research_plan_v3* (zero-inflated, correlation-aware,
device-era-conditioned conditional diffusion for the Kitakyushu 20-year
district integrated energy system dataset).

Generates joint daily scenarios `Y ∈ ℝ^{5×24}` for the five exogenous
uncertainty-source variables `[PV, E, C, H, HW]` conditioned on predicted
weather, calendar, sample-out-of-sample point forecast, observed prefix and
device-era. Supply/result variables and storage are **not** generated (the real
system has no storage; plan C6/C7).

## Layout

```
sc_cdiff/
├── configs/default.yaml          # all paths / hyperparameters / device
├── normalize.py                  # per-channel (PV per-era, C/H activated-only) scaling
├── ema.py                        # weight EMA
├── data/
│   ├── build_dataset.py          # xlsx -> continuous hourly tape (.npz) + norm_stats.json
│   └── dataset.py                # TrainWindows (sliding) / AlignedDays (natural days)
├── forecast/
│   ├── pointforecast.py          # sample-out-of-sample Yhat (climatology, train-only)
│   └── weatherproxy.py           # predicted-weather proxy What + oracle weather
├── models/
│   ├── diffusion.py              # cosine schedule, q_sample / p_step
│   ├── conditioning.py           # shared conditioning encoder h_cond
│   ├── denoiser.py               # CSDI-style two-axis attention eps_theta
│   ├── gate.py                   # zero-inflation gate branch g_phi
│   └── sc_cdiff.py               # assembly + loss (diff/gate/corr/phy) + gated sampler
├── train.py                      # AdamW + EMA, early stop on validation Energy Score
├── sample.py                     # M scenarios per test day for each horizon k
├── eval/                         # marginal / temporal / joint / physical / reliability / decision
└── scripts/run_pipeline.sh
```

## Setup (Linux, RTX 3090 board #2)

```bash
pip install -r sc_cdiff/requirements.txt   # install a CUDA build of torch for the 3090
# place the dataset at the path in configs/default.yaml -> paths.raw_xlsx
#   default: /workspace/code/processed_data.xlsx
```

The GPU is selected by `device: cuda:2` in `configs/default.yaml` (override with
`--device`). It falls back to CPU if CUDA is unavailable.

## Run

```bash
# from the directory that CONTAINS sc_cdiff/ :
bash sc_cdiff/scripts/run_pipeline.sh
# or step by step:
python -m sc_cdiff.data.build_dataset
python -m sc_cdiff.forecast.pointforecast
python -m sc_cdiff.forecast.weatherproxy
python -m sc_cdiff.train          --device cuda:2
python -m sc_cdiff.sample          --device cuda:2 --split test
python -m sc_cdiff.eval.run_eval   --split test
```

All artifacts (dataset.npz, norm_stats.json, yhat/what npy, ckpt_best.pt,
scenarios_*.npz, eval_*.json) go to `paths.artifacts`.

## Verified data facts (against the plan)

`build_dataset` prints sanity checks that match the plan's verified numbers:
187752 hourly rows → 7823 days = **6058 / 730 / 1035** (train/val/test); device
eras **3835 / 1705 / 2283** days; **31** earthquake-filled days (2011-03);
zero ratios C 74.0% / H 71.9% / HW 0.6%; activated mean/std C 807/803, H
586/555, HW 70/40.

### Missing-data handling (plan C8)

`processed_data.xlsx` still contains the 2011-03 earthquake gap (744 NaN hours
in PV, E, temperature, wind direction). The builder imputes them with
(month,hour) climatology computed from training rows, flags them in
`FILLEDfull`, **skips any training window overlapping them**, and **excludes
them from evaluation**.

### Leakage control (plan §8.2)

- `Yhat`: climatology point forecaster fit on TRAIN only, never noised truth.
- `What`: (month,hour) weather climatology proxy with realistic error; true
  weather is saved separately for the oracle upper-bound ablation only
  (`use_oracle_weather: true` in config).

## Key design points

- **Sliding-window augmentation** (training only; eval uses aligned natural days).
- **Zero-inflation**: PV night gating from true irradiance (physical, exact);
  C/H Bernoulli activation gate sampled then multiplied by the diffusion
  amplitude; E/HW non-negativity.
- **Two-axis attention** denoiser carries cross-variable dependence.
- **Loss** = `L_diff + 1.0·L_gate + 0.1·L_corr + 0.05·L_phy`, where `L_corr`
  is a per-season correlation regularizer on the Tweedie one-step `x0` estimate.
- **Ablations**: set loss weights to 0 (`lambda_gate`/`lambda_corr`) or
  `use_oracle_weather` for the oracle bound. Device-era ablation: set all
  `ERAfull` to a constant in a custom dataset build.

Model is ~1.3 M parameters (`d_model=128, L=4`); trains fast on the 3090.
