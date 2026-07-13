#!/usr/bin/env bash
# One-shot: merge datasets -> train NsDiff on IES -> figures + metrics (GPU 4)
set -e
cd "$(dirname "$0")"
export PYTHONPATH=./

# 1) merge Total_energy.csv + Total_weather.csv -> data/IES/IES.csv
python3 data_merge.py --dir ./data/IES

# 2) train + evaluate on GPU 4
bash ./scripts/NSDiff/IES.sh
