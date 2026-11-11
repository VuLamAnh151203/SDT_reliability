#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One freely learned prototype bank per modality (p_c^T, p_c^A, p_c^V),
# aligned to freely learned shared semantic centers q_c.  Only q_c receives
# the normalized Emotion-Wheel structure constraint.
LAMBDA_PROTO="${LAMBDA_WHEEL_PROTO:-0.1}"
LAMBDA_STRUCTURE="${LAMBDA_WHEEL_STRUCTURE:-0.05}"
LAMBDA_ALIGNMENT="${LAMBDA_WHEEL_ALIGNMENT:-0.05}"
LAMBDA_BOUNDARY="${LAMBDA_WHEEL_BOUNDARY:-0.01}"
LAMBDA_CPCC="${LAMBDA_WHEEL_CPCC:-0}"
HYPERBOLIC_DIM="${WHEEL_HYPERBOLIC_DIM:-16}"
MAX_RADIUS="${WHEEL_FREE_PROTOTYPE_MAX_RADIUS:-0.90}"

export TICAL_MODE=observe
exec bash "$SCRIPT_DIR/exec_iemocap_tical.sh" \
  --use-emotion-wheel \
  --wheel-geometry poincare \
  --wheel-feature-source pure \
  --wheel-prototype-layout modality-specific \
  --wheel-prototype-radius-mode fixed-shared \
  --wheel-prototype-radius 0.75 \
  --wheel-free-prototype-max-radius "$MAX_RADIUS" \
  --hyperbolic-dim "$HYPERBOLIC_DIM" \
  --wheel-temperature 1.0 \
  --lambda-wheel-proto "$LAMBDA_PROTO" \
  --lambda-wheel-structure "$LAMBDA_STRUCTURE" \
  --lambda-wheel-alignment "$LAMBDA_ALIGNMENT" \
  --lambda-wheel-boundary "$LAMBDA_BOUNDARY" \
  --lambda-wheel-cpcc "$LAMBDA_CPCC" \
  --lambda-wheel-radius 0 \
  --lambda-hyp 0 \
  "$@"
