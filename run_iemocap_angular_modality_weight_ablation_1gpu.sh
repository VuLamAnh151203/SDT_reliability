#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-all}"
if (( $# > 0 )); then
  shift
fi

case "$VARIANT" in
  uniform)
    EXPERIMENTS=("uniform|uniform|1 1 1")
    ;;
  fixed1|40-35-25)
    EXPERIMENTS=("fixed_40_35_25|fixed|0.40 0.35 0.25")
    ;;
  fixed2|45-35-20)
    EXPERIMENTS=("fixed_45_35_20|fixed|0.45 0.35 0.20")
    ;;
  fixed3|50-35-15)
    EXPERIMENTS=("fixed_50_35_15|fixed|0.50 0.35 0.15")
    ;;
  adaptive)
    EXPERIMENTS=("adaptive|adaptive|1 1 1")
    ;;
  all)
    EXPERIMENTS=(
      "uniform|uniform|1 1 1"
      "fixed_40_35_25|fixed|0.40 0.35 0.25"
      "fixed_45_35_20|fixed|0.45 0.35 0.20"
      "fixed_50_35_15|fixed|0.50 0.35 0.15"
      "adaptive|adaptive|1 1 1"
    )
    ;;
  *)
    echo "Usage: $0 {uniform|fixed1|fixed2|fixed3|adaptive|all} [train.py arguments...]" >&2
    exit 2
    ;;
esac

ROOT_OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/angular_modality_weight_ablation_1gpu}"
for experiment in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r label mode weights <<< "$experiment"
  echo "=== Prototype modality weighting: $label ==="
  PROTOTYPE_MODALITY_WEIGHT_MODE="$mode" \
  PROTOTYPE_MODALITY_WEIGHTS="$weights" \
  EXPERIMENT_SCRIPT=exec_iemocap_angular_modality_weights.sh \
  OUTPUT_DIR="$ROOT_OUTPUT_DIR/$label" \
    bash "$SCRIPT_DIR/run_iemocap_freeproto_interactive_1gpu.sh" "$@"
done
