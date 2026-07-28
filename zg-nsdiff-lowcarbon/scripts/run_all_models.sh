#!/usr/bin/env bash
# Full experiment: data prep -> DeepVAR -> D3U -> WaveStitch -> NsDiff
set -e
source "$(dirname "$0")/_env.sh"
bash scripts/prepare_heew.sh
bash scripts/run_deepvar.sh
bash scripts/run_d3u.sh
bash scripts/run_wavestitch.sh
bash scripts/run_nsdiff.sh
