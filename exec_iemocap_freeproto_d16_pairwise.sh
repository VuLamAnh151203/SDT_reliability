#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Variant 2: variant 1 + normalized pairwise Wheel-distance MSE.
export PROTOTYPE_LAYOUT=free
export STRUCTURE_MODE=pairwise-mse
export LAMBDA_STRUCTURE="${LAMBDA_WHEEL_STRUCTURE:-0.05}"
export LAMBDA_PRIVATE=0
exec bash "$SCRIPT_DIR/exec_iemocap_freeproto_d16_ablation_base.sh" "$@"
