#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_SCRIPT=exec_iemocap_wheel_learned_prototypes_from_start.sh \
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/learned_prototypes_from_start_1gpu}" \
  exec bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
