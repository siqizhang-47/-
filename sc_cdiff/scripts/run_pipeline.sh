#!/usr/bin/env bash
# End-to-end SC-CDiff pipeline. Run from the directory that CONTAINS sc_cdiff/
# (i.e. `python -m sc_cdiff.*` must resolve). GPU = RTX 3090 board index 2.
set -euo pipefail

CFG=${CFG:-sc_cdiff/configs/default.yaml}
DEVICE=${DEVICE:-cuda:2}

echo "==> [1/6] build dataset + norm stats"
python -m sc_cdiff.data.build_dataset --config "$CFG"

echo "==> [2/6] point forecast (Yhat)"
python -m sc_cdiff.forecast.pointforecast --config "$CFG"

echo "==> [3/6] weather proxy (What) + oracle"
python -m sc_cdiff.forecast.weatherproxy --config "$CFG"

echo "==> [4/6] train (device=$DEVICE)"
python -m sc_cdiff.train --config "$CFG" --device "$DEVICE"

echo "==> [5/6] sample test scenarios (all k)"
python -m sc_cdiff.sample --config "$CFG" --device "$DEVICE" --split test

echo "==> [6/6] evaluate"
python -m sc_cdiff.eval.run_eval --config "$CFG" --split test

echo "Done. Artifacts in the configured paths.artifacts directory."
