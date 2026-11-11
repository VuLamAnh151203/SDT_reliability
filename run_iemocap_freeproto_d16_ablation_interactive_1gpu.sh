#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-all}"
if (( $# > 0 )); then
  shift
fi

case "$VARIANT" in
  1|ce|ce-only)
    RUNNERS=(exec_iemocap_freeproto_d16_ce_only.sh)
    ;;
  2|pairwise|pairwise-mse)
    RUNNERS=(exec_iemocap_freeproto_d16_pairwise.sh)
    ;;
  3|ordinal)
    RUNNERS=(exec_iemocap_freeproto_d16_ordinal.sh)
    ;;
  4|shared-private)
    RUNNERS=(exec_iemocap_freeproto_d16_ordinal_shared_private.sh)
    ;;
  all)
    RUNNERS=(
      exec_iemocap_freeproto_d16_ce_only.sh
      exec_iemocap_freeproto_d16_pairwise.sh
      exec_iemocap_freeproto_d16_ordinal.sh
      exec_iemocap_freeproto_d16_ordinal_shared_private.sh
    )
    ;;
  *)
    echo "Usage: $0 {1|2|3|4|all} [train.py arguments...]" >&2
    exit 2
    ;;
esac

ROOT_OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/freeproto_d16_ablation_1gpu}"
for runner in "${RUNNERS[@]}"; do
  label="${runner#exec_iemocap_freeproto_d16_}"
  label="${label%.sh}"
  echo "=== Running D16 ablation: $label ==="
  EXPERIMENT_SCRIPT="$runner" \
  OUTPUT_DIR="$ROOT_OUTPUT_DIR/$label" \
    bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
done
