#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 1: modality-specific p_c^T/p_c^A/p_c^V, D=32, no logit fusion.
export WHEEL_HYPERBOLIC_DIM=32
export LAMBDA_WHEEL_ALIGNMENT="${LAMBDA_WHEEL_ALIGNMENT:-0.05}"
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_modality_prototypes.sh" \
  --prototype-evidence-fusion none \
  --prototype-evidence-weight 0 \
  --wheel-prototypes-per-class 1 \
  --prototype-loss-warmup-epochs 0 \
  --prototype-loss-ramp-epochs 0 \
  "$@"
