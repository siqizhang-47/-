#!/usr/bin/env bash
# Train + sample + evaluate every baseline, then collect the master table.
# Run from the directory containing sc_cdiff/. Assumes build_dataset / forecasts
# have already been produced. GPU = cuda:2 by default.
set -euo pipefail
CFG=${CFG:-sc_cdiff/configs/default.yaml}
DEVICE=${DEVICE:-cuda:2}
SPLIT=${SPLIT:-test}

echo "==> statistical baselines (no training)"
for m in historical weather_knn copula; do
  python -m sc_cdiff.baselines.statistical --config "$CFG" --method "$m" --split "$SPLIT"
  python -m sc_cdiff.eval.run_eval --config "$CFG" --split "$SPLIT" --tag "$m"
done

echo "==> CSDI baseline (SC-CDiff backbone, no gate/era/corr)"
python -m sc_cdiff.train        --config "$CFG" --device "$DEVICE" --tag csdi \
       --disable_gate --disable_era --lambda_corr 0
python -m sc_cdiff.sample       --config "$CFG" --device "$DEVICE" --tag csdi \
       --disable_gate --disable_era --split "$SPLIT"
python -m sc_cdiff.eval.run_eval --config "$CFG" --split "$SPLIT" --tag csdi

echo "==> SSSD and TimeGrad diffusion baselines"
for m in sssd timegrad; do
  python -m sc_cdiff.baselines.run_diffusion --config "$CFG" --device "$DEVICE" \
         --model "$m" --split "$SPLIT" --stage both
  python -m sc_cdiff.eval.run_eval --config "$CFG" --split "$SPLIT" --tag "$m"
done

echo "==> GAN baselines (WGAN, conditional WGAN-GP)"
for m in wgan cwgan_gp; do
  python -m sc_cdiff.baselines.gan --config "$CFG" --device "$DEVICE" \
         --method "$m" --split "$SPLIT" --stage both
  python -m sc_cdiff.eval.run_eval --config "$CFG" --split "$SPLIT" --tag "$m"
done

echo "==> collect master table"
for k in 0 6 12 18; do
  python -m sc_cdiff.eval.collect_results --config "$CFG" --split "$SPLIT" --k "$k" || true
done
