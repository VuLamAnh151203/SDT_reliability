#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Variant 3: variant 1 + ordinal Wheel ranking constraints.
export PROTOTYPE_LAYOUT=free
export STRUCTURE_MODE=ordinal
export LAMBDA_STRUCTURE="${LAMBDA_WHEEL_STRUCTURE:-0.05}"
export LAMBDA_PRIVATE=0
exec bash "$SCRIPT_DIR/exec_iemocap_freeproto_d16_ablation_base.sh" "$@"
