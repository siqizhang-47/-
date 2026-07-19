# Third-party provenance

## NsDiff (upstream reference)

- Source: user-provided archive `NsDiff-main.zip` (official ICML 2025 repo,
  "Non-stationary Diffusion For Probabilistic Time Series Forecasting").
- Archive SHA-256 (recorded at preprocessing time into
  `artifacts/reproducibility/source_manifest.json`):
  `ead2489b57db0d16044d30102d558d3e2e882f4b87d23b21685b6f41e9dbaa5c`
- Usage: the original files are NOT modified. The diffusion mathematics
  (`make_beta_schedule`, `compute_tilde_alpha`, `compute_hat_alpha`,
  `cal_forward_noise`, `cal_sigma_tilde`, `calc_gammas`, `q_sample`,
  `p_sample*`) and the mu/g backbone structure were copied into
  `src/models/diffusion_utils.py`, `src/layer/mu_backbone_exo.py`,
  `src/layer/g_backbone_exo.py`, `src/layer/denoise_exo.py`, with the
  documented changes: exogenous conditioning, per-coordinate losses,
  numerical clamps, chunked sampling, corrected metrics.

## D3U adapter

`src/baselines/d3u_adapter.py` is a self-contained re-implementation of the
D3U decomposition (deterministic backbone + diffusion over the point-forecast
residual), adapted to the common low-carbon data interface with oracle future
weather. It does not vendor upstream D3U code. If the official implementation
is later vendored under `third_party/D3U/`, record source / commit /
environment / local patches / run command here.

## WaveStitch adapter

`src/baselines/wavestitch_adapter.py` is a self-contained re-implementation
of WaveStitch-style conditional sequence completion (mask-conditioned
diffusion inpainting with RePaint-style history imposition), adapted to the
same interface. Same vendoring rule as above.
