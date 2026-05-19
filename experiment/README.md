# EWELD Extreme-Weather Load Forecasting Experiment

End-to-end implementation of the experimental protocol described in the
task document, comparing three time-series-library style baselines on the
EWELD `EWELD_labeled_output` dataset:

| Baseline      | Source                                  |
|---------------|------------------------------------------|
| iTransformer  | `baselines/iTransformer-main/`           |
| TimeMixer     | `baselines/TimeMixer-main/`              |
| TimeXer       | `baselines/TimeXer-main/`                |

The unified experiment driver (`experiment/`) replaces each baseline's
per-dataset CSV loader with a single load → filter → feature → window →
normalize → split pipeline that conforms to the spec; the baselines'
original `Model` classes are kept untouched.

## Data assumed on disk

```
EWELD_labeled_output/
├── CT1/
│   ├── C10 Manufacture of food products/
│   │   ├── U14.csv
│   │   └── ...
│   └── ...
├── CT2/
└── CT3/
```

Each `U*.csv` has 15-min sampled columns:
`Time, Load, User, Temperature, Dew Point, Humidity, Wind, Wind Speed,
Wind Gust, Pressure, Condition, 01Low_te, ..., 20Light_S`.

The pipeline drops `User`, `Wind`, `Condition`, and extreme-weather labels
13–20, keeps labels 01–12 only, and adds cyclic time features
(`slot_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend`).

## High-quality user filter

A user is kept iff:

- ≥ 1 year of 15-min samples (≥ 35,040 rows)
- missing-load ratio < 5%
- chronological 70 / 10 / 20 split each holds ≥ `seq_len + pred_len` rows
- the test split contains ≥ 30 prediction windows with any 01–12 label = 1
- non-zero load ratio > 80%

> Note: the spec document gives **70% / 10% / 30%** (sums to 110%). We
> default to **70 / 10 / 20** — the most likely intended split. The
> `--train_ratio / --val_ratio / --test_ratio` CLI flags let you change it.

## Normalization

- **Load** — per-user z-score, statistics fit on the user's train slice.
- **Weather** (6 cols) — per-city (`CT1` / `CT2` / `CT3`) z-score, fit on
  all users in the city using their train slices.
- Event labels (01–12) and time features are left unscaled.

## Windowing

- `seq_len = 96` (lookback 24 h) — fixed.
- `pred_len ∈ {24, 48, 96}` (6 h / 12 h / 24 h) — horizon ablation.
- Train stride 4 (≈ one window per hour), val/test stride 1.
- Channel order: `[<exogenous>..., slot_sin, ..., is_weekend, e01, ..., e12, Load]`
  — `Load` is **last**, matching `features='MS'` in the three baselines.
- Train / val / test datasets across users are pooled into a single
  multi-user dataset.

## Event-window classification

For each prediction window `[t, t+H-1]` and each user, we use the raw
binary labels in that interval:

- `EventWindow(t) = 1` iff any of the 12 kept labels is set anywhere in
  the horizon.
- `NormalWindow(t) = 1 - EventWindow(t)`.
- Family flags: `temp` (01–04), `wind` (05–07), `typhoon` (08–12).

Labels 13–20 do **not** participate in either window or family definition.

## Metrics

Computed on inverse-normalized real load (original kW units):

- **Overall:** MAE, RMSE, sMAPE, P95AE, MAE_peak
- **Normal-only window:** MAE_normal, RMSE_normal, sMAPE_normal
- **Event window:** MAE_event, RMSE_event, sMAPE_event, P95AE_event, MAE_peak_event
- **Per family (temp/wind/typhoon):** MAE, RMSE, sMAPE

`experiment/aggregate.py` collects every `results/*.json` and writes two
CSV summary tables (`summary_main.csv`, `summary_family.csv`) mirroring
the layout in the spec.

## Running

The target GPU is **device 2** (NVIDIA GeForce RTX 3090, CUDA 12.9). Set
the data path through `DATA_ROOT`.

Default `DATA_ROOT` (and the `--data_root` CLI default) is
`/workspace/data/sxq_data/EWELD_labeled_output`. Override with the
`DATA_ROOT=` env var or `--data_root` flag if your data lives elsewhere.

```bash
# main: three baselines × full feature set × H=96
GPU=2 ./scripts/run_main.sh

# ablation 1: weather / event feature contribution
GPU=2 ./scripts/run_ablation_features.sh

# ablation 2: horizon length
GPU=2 ./scripts/run_ablation_horizon.sh
```

Or invoke a single configuration manually:

```bash
CUDA_VISIBLE_DEVICES=2 python -m experiment.run \
    --model iTransformer \
    --feature_set load_weather_event \
    --pred_len 96 \
    --epochs 10 \
    --gpu 2
```

Logs go to `logs/<model>_*.log`, per-run JSON to `results/*.json`.

## Hyperparameters

Reasonable defaults for the three baselines (`experiment/models/factory.py`):

- `d_model=128`, `n_heads=8`, `e_layers=2`, `d_ff=256`, `dropout=0.1`
- iTransformer: `use_norm=True`, `class_strategy='projection'`
- TimeMixer: `channel_independence=1`, `down_sampling_layers=2`, `down_sampling_window=2`, `moving_avg=25`
- TimeXer: `patch_len=16`, `use_norm=True`

Optimizer is Adam(lr=1e-3) with MSE loss, gradient clipping at 5.0, and
early stopping on validation MSE (patience 3).

## Environment

```bash
conda env create -f environment.yml
conda activate eweld
# or:
pip install -r requirements.txt
# plus a CUDA-matched PyTorch wheel:
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0
```

CUDA driver 12.9 on the host is forward-compatible with the cu121 build
of PyTorch 2.4.
