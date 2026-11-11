#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "Usage: GPU_SLOT=0 bash $0 {1|2|3|4|5|6} [train.py arguments...]" >&2
  exit 2
fi
shift

case "$STAGE" in
  1) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage1_modality_d32.sh" ;;
  2) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage2_detached.sh" ;;
  3) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage3_loss_ramp.sh" ;;
  4) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage4_j2.sh" ;;
  5) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage5_j2_barycenter.sh" ;;
  6) EXPERIMENT_SCRIPT="exec_iemocap_proto_stage6_supervised_selector.sh" ;;
  *) echo "Error: stage must be one of 1,2,3,4,5,6; got '$STAGE'." >&2; exit 2 ;;
esac

export EXPERIMENT_SCRIPT
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/prototype_stage${STAGE}_1gpu}"
exec bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
