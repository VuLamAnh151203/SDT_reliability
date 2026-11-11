#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Shared controls for the D=16 prototype ablation.  Every free bank starts
# isotropically in all 16 tangent dimensions, so no variant inherits the old
# fixed 2-D Wheel coordinates.  The four wrappers below change only layout and
# structural supervision.
PROTOTYPE_LAYOUT="${PROTOTYPE_LAYOUT:-free}"
STRUCTURE_MODE="${STRUCTURE_MODE:-pairwise-mse}"
LAMBDA_STRUCTURE="${LAMBDA_STRUCTURE:-0}"
LAMBDA_PRIVATE="${LAMBDA_PRIVATE:-0}"

export TICAL_MODE=observe
exec bash "$SCRIPT_DIR/exec_iemocap_tical.sh" \
  --use-emotion-wheel \
  --wheel-geometry poincare \
  --wheel-feature-source pure \
  --wheel-prototype-layout "$PROTOTYPE_LAYOUT" \
  --wheel-free-prototype-init random \
  --wheel-prototype-structure-mode "$STRUCTURE_MODE" \
  --wheel-ordinal-margin "${WHEEL_ORDINAL_MARGIN:-0.2}" \
  --wheel-private-residual-init-std "${WHEEL_PRIVATE_INIT_STD:-0.02}" \
  --wheel-prototype-radius-mode fixed-shared \
  --wheel-prototype-radius 0.75 \
  --wheel-free-prototype-max-radius "${WHEEL_FREE_PROTOTYPE_MAX_RADIUS:-0.90}" \
  --hyperbolic-dim 16 \
  --wheel-temperature 1.0 \
  --lambda-wheel-proto "${LAMBDA_WHEEL_PROTO:-0.1}" \
  --lambda-wheel-structure "$LAMBDA_STRUCTURE" \
  --lambda-wheel-private-residual "$LAMBDA_PRIVATE" \
  --lambda-wheel-boundary "${LAMBDA_WHEEL_BOUNDARY:-0}" \
  --lambda-wheel-cpcc 0 \
  --lambda-wheel-radius 0 \
  --lambda-wheel-alignment 0 \
  --lambda-wheel-diversity 0 \
  --lambda-wheel-utilization 0 \
  --lambda-hyp 0 \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  "$@"
