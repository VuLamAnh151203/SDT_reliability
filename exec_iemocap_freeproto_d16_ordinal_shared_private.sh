#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Variant 4: ordinal Wheel supervision on shared class centers plus zero-mean
# modality-private residuals used by the Prototype CE branch.
export PROTOTYPE_LAYOUT=shared-private
export STRUCTURE_MODE=ordinal
export LAMBDA_STRUCTURE="${LAMBDA_WHEEL_STRUCTURE:-0.05}"
export LAMBDA_PRIVATE="${LAMBDA_WHEEL_PRIVATE_RESIDUAL:-0.01}"
exec bash "$SCRIPT_DIR/exec_iemocap_freeproto_d16_ablation_base.sh" "$@"
