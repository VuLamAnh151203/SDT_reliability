#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This wrapper intentionally reuses the allocation-safe one-GPU launcher.
# It never starts, cancels, or replaces the current Slurm interactive session.
export EXPERIMENT_SCRIPT=exec_iemocap_angular_relational_prototypes.sh
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/angular_relational_1gpu}"

exec bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
