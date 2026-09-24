#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXPERIMENT="${EXPERIMENT^^}"
DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
FIXED_RADIUS="${FIXED_RADIUS:-0.75}"
CAUSAL_OUTPUT_DIR="${CAUSAL_OUTPUT_DIR:-$SCRIPT_DIR/results/causal_${DATASET,,}}"

ALL_IDS=(P-FREE P-FIXED P-2D P-ZERORES E-FREE)
if [[ "$EXPERIMENT" == "ALL" ]]; then
  IDS=("${ALL_IDS[@]}")
else
  IDS=("$EXPERIMENT")
fi

run_one() {
  local id="$1"
  shift
  local base_id
  local constraints=()
  case "$id" in
    P-FREE)
      base_id="P-PC"
      constraints=(--hyperbolic-dim 16 --wheel-radius-mode free)
      ;;
    P-FIXED)
      base_id="P-PC"
      constraints=(--hyperbolic-dim 16 --wheel-radius-mode fixed
                   --wheel-fixed-radius "$FIXED_RADIUS")
      ;;
    P-2D)
      base_id="P-PC"
      constraints=(--hyperbolic-dim 2 --wheel-radius-mode free)
      ;;
    P-ZERORES)
      base_id="P-PC"
      constraints=(--hyperbolic-dim 16 --wheel-radius-mode free
                   --wheel-zero-residual)
      ;;
    E-FREE)
      base_id="E-PC"
      constraints=(--hyperbolic-dim 16 --wheel-radius-mode free)
      ;;
    *)
      echo "Unknown experiment '$id'. Use: ${ALL_IDS[*]} or all" >&2
      exit 2
      ;;
  esac

  echo "#################################################################"
  echo "Causal geometry experiment: $id"
  echo "#################################################################"
  DATASET="$DATASET" SEEDS="$SEEDS" OUTPUT_DIR="$CAUSAL_OUTPUT_DIR" \
    bash "$SCRIPT_DIR/run_ablation.sh" "$base_id" \
      "${constraints[@]}" "$@"
}

for id in "${IDS[@]}"; do
  run_one "$id" "$@"
done

echo "Summarize after all requested runs finish:"
echo "  ${PYTHON:-python} $SCRIPT_DIR/summarize_causal.py $CAUSAL_OUTPUT_DIR"
