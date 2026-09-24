#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Original SDT losses + Wheel Prototype + Wheel CPCC only.
# observe disables CA-KD, HypCPCC, and TiCAL fusion adjustment.
exec bash "$SCRIPT_DIR/exec_iemocap_tical_wheel.sh" \
  --tical-mode observe \
  --hyperbolic-dim 16 \
  --lambda-hyp 0 \
  --lambda-wheel-proto 0.1 \
  --lambda-wheel-cpcc 0.05 \
  "$@"
