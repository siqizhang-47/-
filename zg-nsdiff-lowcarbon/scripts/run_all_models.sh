#!/usr/bin/env bash
# Full experiment: data prep -> NsDiff -> ZG-NsDiff -> D3U -> WaveStitch
set -e
source "$(dirname "$0")/_env.sh"
bash scripts/prepare_low_carbon.sh
bash scripts/pretrain_nsdiff_low_carbon.sh
bash scripts/run_nsdiff_low_carbon.sh
bash scripts/run_zg_nsdiff_low_carbon.sh
bash scripts/run_d3u_low_carbon.sh
bash scripts/run_wavestitch_low_carbon.sh
