#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Learn one fully free D-dimensional prototype per emotion in the Poincare
# ball.  The prototypes start from the semantic Wheel only for stable
# initialization; after that every coordinate and radius is trainable.
#
# The structure loss constrains only normalized pairwise prototype distances,
# so absolute position, orientation, global scale and residual dimensions stay
# data-driven.  Feature-level Wheel CPCC is off by default to isolate this
# prototype-learning mechanism; set LAMBDA_WHEEL_CPCC to enable it as an
# additional ablation.
LAMBDA_PROTO="${LAMBDA_WHEEL_PROTO:-0.1}"
LAMBDA_STRUCTURE="${LAMBDA_WHEEL_STRUCTURE:-0.05}"
LAMBDA_BOUNDARY="${LAMBDA_WHEEL_BOUNDARY:-0.01}"
LAMBDA_CPCC="${LAMBDA_WHEEL_CPCC:-0}"
HYPERBOLIC_DIM="${WHEEL_HYPERBOLIC_DIM:-16}"
MAX_RADIUS="${WHEEL_FREE_PROTOTYPE_MAX_RADIUS:-0.90}"

export TICAL_MODE=observe
exec bash "$SCRIPT_DIR/exec_iemocap_tical.sh" \
  --use-emotion-wheel \
  --wheel-geometry poincare \
  --wheel-feature-source pure \
  --wheel-prototype-layout free \
  --wheel-prototype-radius-mode fixed-shared \
  --wheel-prototype-radius 0.75 \
  --wheel-free-prototype-max-radius "$MAX_RADIUS" \
  --hyperbolic-dim "$HYPERBOLIC_DIM" \
  --wheel-temperature 1.0 \
  --lambda-wheel-proto "$LAMBDA_PROTO" \
  --lambda-wheel-structure "$LAMBDA_STRUCTURE" \
  --lambda-wheel-boundary "$LAMBDA_BOUNDARY" \
  --lambda-wheel-cpcc "$LAMBDA_CPCC" \
  --lambda-wheel-radius 0 \
  --lambda-hyp 0 \
  "$@"
