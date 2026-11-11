#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_SCRIPT="exec_iemocap_wheel_modality_prototypes.sh"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/modality_proto_1gpu}"
exec bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
