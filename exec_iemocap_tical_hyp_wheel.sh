#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# SDT + CA-KD + HypCPCC + Wheel Prototype + Wheel CPCC.
# tical-mode=hyp deliberately excludes TiCAL fusion adjustment.
exec bash "$SCRIPT_DIR/exec_iemocap_tical_wheel.sh" \
  --tical-mode hyp \
  --tical-warmup-epochs 10 \
  --hyperbolic-dim 16 \
  --wheel-prototype-radius 0.75 \
  --wheel-temperature 1.0 \
  --wheel-anchor-mix 0.5 \
  --lambda-hyp 0.1 \
  --lambda-wheel-proto 0.1 \
  --lambda-wheel-cpcc 0.05 \
  "$@"
