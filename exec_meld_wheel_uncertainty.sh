#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# SDT + Poincare Emotion Wheel + Prototype/CPCC + uncertainty-aware fusion.
# Usage:
#   bash exec_meld_wheel_uncertainty.sh [none|gate|supervise|full] --gpu-id 0
# The default is the complete proposed mode.
MODE="${WHEEL_UNCERTAINTY_MODE:-full}"
if [[ $# -gt 0 ]]; then
  case "$1" in
    none|gate|supervise|full)
      MODE="$1"
      shift
      ;;
  esac
fi

case "$MODE" in
  none|gate)
    LAMBDA_UNCERTAINTY="0"
    ;;
  supervise|full)
    LAMBDA_UNCERTAINTY="${LAMBDA_WHEEL_UNCERTAINTY:-0.1}"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use none, gate, supervise, or full." >&2
    exit 2
    ;;
esac

exec bash "$SCRIPT_DIR/exec_meld.sh" \
  --use-tical --tical-mode observe \
  --use-emotion-wheel --wheel-geometry poincare \
  --wheel-radius-mode free \
  --hyperbolic-dim 16 \
  --wheel-prototype-radius 0.75 \
  --wheel-temperature 1.0 \
  --lambda-wheel-proto 0.1 \
  --lambda-wheel-cpcc 0.05 \
  --wheel-uncertainty-mode "$MODE" \
  --lambda-wheel-uncertainty "$LAMBDA_UNCERTAINTY" \
  --wheel-uncertainty-hidden-dim 32 \
  --wheel-uncertainty-temperature 1.0 \
  --wheel-uncertainty-gate-strength 0.5 \
  --wheel-uncertainty-warmup-epochs 10 \
  --wheel-uncertainty-ramp-epochs 5 \
  --lambda-hyp 0 \
  --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 \
  "$@"
