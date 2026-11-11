#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the allocation/Python/GPU validation from the existing one-GPU runner.
EXPERIMENT_SCRIPT=exec_iemocap_wheel_dual_data_prototypes.sh \
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/dual_data_prototypes_1gpu}" \
  exec bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
