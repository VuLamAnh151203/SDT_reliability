#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Variant 1: free shared D16 prototypes + supervised Prototype CE.
# No pairwise/ordinal/CPCC Wheel constraint is active.
export PROTOTYPE_LAYOUT=free
export STRUCTURE_MODE=pairwise-mse
export LAMBDA_STRUCTURE=0
export LAMBDA_PRIVATE=0
exec bash "$SCRIPT_DIR/exec_iemocap_freeproto_d16_ablation_base.sh" "$@"
