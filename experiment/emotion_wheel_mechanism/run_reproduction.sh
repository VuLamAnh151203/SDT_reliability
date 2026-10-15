#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

ACTION="${1:-ablation}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/results/reproduction}"
REFERENCE_DIR="$RUN_ROOT/reference_ablation"
GEOMETRY_DIR="$RUN_ROOT/full_geometry_ablation"
CONTROL_DIR="$RUN_ROOT/mechanism_controls"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-test}"
VALID_RATIO="${VALID_RATIO:-0.1}"

case "$SELECTION_PROTOCOL" in
  test|validation) ;;
  *)
    echo "SELECTION_PROTOCOL must be test or validation; got: $SELECTION_PROTOCOL" >&2
    exit 2
    ;;
esac

# Keep analysis on the same GPU when the caller supplies --gpu-id as a
# training argument instead of exporting GPU_ID.
TRAIN_ARGS=("$@")
for ((arg_index = 0; arg_index < ${#TRAIN_ARGS[@]}; arg_index++)); do
  case "${TRAIN_ARGS[$arg_index]}" in
    --gpu-id)
      if ((arg_index + 1 >= ${#TRAIN_ARGS[@]})); then
        echo "--gpu-id requires a value" >&2
        exit 2
      fi
      GPU_ID="${TRAIN_ARGS[$((arg_index + 1))]}"
      ;;
    --gpu-id=*)
      GPU_ID="${TRAIN_ARGS[$arg_index]#--gpu-id=}"
      ;;
  esac
done

print_layout() {
  echo "Reproduction root : $RUN_ROOT"
  echo "Reference runs    : $REFERENCE_DIR"
  echo "Full geometry runs: $GEOMETRY_DIR"
  echo "Mechanism controls: $CONTROL_DIR"
  echo "Dataset           : $DATASET"
  echo "Seeds             : $SEEDS"
  echo "Checkpoint choice : $SELECTION_PROTOCOL weighted F1"
  if [[ "$SELECTION_PROTOCOL" == "validation" ]]; then
    echo "Validation ratio  : $VALID_RATIO"
  fi
}

run_reference_ablation() {
  mkdir -p "$REFERENCE_DIR"
  local id
  for id in B0 P-P P-C P-PC; do
    DATASET="$DATASET" SEEDS="$SEEDS" OUTPUT_DIR="$REFERENCE_DIR" \
      SELECTION_PROTOCOL="$SELECTION_PROTOCOL" VALID_RATIO="$VALID_RATIO" \
      bash "$ROOT_DIR/geometry_analysis/run_ablation.sh" "$id" \
      --fusion-feature-source enhanced \
      --tical-warmup-epochs 10 \
      --wheel-radius-mode free \
      --wheel-semantic-control correct \
      --wheel-uncertainty-mode none \
      --lambda-wheel-uncertainty 0 \
      "$@"
  done
}

run_full_geometry_ablation() {
  mkdir -p "$GEOMETRY_DIR"
  DATASET="$DATASET" SEEDS="$SEEDS" OUTPUT_DIR="$GEOMETRY_DIR" \
    SELECTION_PROTOCOL="$SELECTION_PROTOCOL" VALID_RATIO="$VALID_RATIO" \
    bash "$ROOT_DIR/geometry_analysis/run_ablation.sh" all \
    --fusion-feature-source enhanced \
    --tical-warmup-epochs 10 \
    --wheel-radius-mode free \
    --wheel-semantic-control correct \
    --wheel-uncertainty-mode none \
    --lambda-wheel-uncertainty 0 \
    "$@"
}

run_mechanism_controls() {
  mkdir -p "$CONTROL_DIR"
  DATASET="$DATASET" SEEDS="$SEEDS" OUTPUT_DIR="$CONTROL_DIR" \
    SELECTION_PROTOCOL="$SELECTION_PROTOCOL" VALID_RATIO="$VALID_RATIO" \
    bash "$SCRIPT_DIR/run_controls.sh" all "$@"
}

run_analysis() {
  local analysis_reference="$REFERENCE_DIR"
  if ! find "$analysis_reference" -type f -name best_checkpoint.pt -print -quit \
      2>/dev/null | grep -q .; then
    analysis_reference="$GEOMETRY_DIR"
  fi
  if ! find "$analysis_reference" -type f -name best_checkpoint.pt -print -quit \
      2>/dev/null | grep -q .; then
    echo "No reference checkpoints found under either:" >&2
    echo "  $REFERENCE_DIR" >&2
    echo "  $GEOMETRY_DIR" >&2
    echo "Run the ablation or geometry stage first." >&2
    exit 2
  fi
  if [[ ! -d "$CONTROL_DIR" ]]; then
    echo "Control directory is missing: $CONTROL_DIR" >&2
    echo "Run the controls stage first." >&2
    exit 2
  fi

  DEVICE="$DEVICE" GPU_ID="$GPU_ID" \
    bash "$SCRIPT_DIR/run_analysis.sh" "$CONTROL_DIR" "$analysis_reference"

  echo "Probing B0/P-P/P-C/P-PC representations..."
  "${PYTHON:-python}" -u "$SCRIPT_DIR/probe_representations.py" \
    "$analysis_reference" --device "$DEVICE" --gpu-id "$GPU_ID" \
    --output "$CONTROL_DIR/mechanism_summary/reference_representation_probes.csv"
}

print_layout
case "$ACTION" in
  ablation)
    run_reference_ablation "$@"
    echo "Reference ablation complete. Next run:"
    echo "  RUN_ROOT='$RUN_ROOT' SEEDS='$SEEDS' GPU_ID='$GPU_ID' bash '$SCRIPT_DIR/run_reproduction.sh' controls --gpu-id '$GPU_ID'"
    ;;
  geometry)
    run_full_geometry_ablation "$@"
    echo "Full B0/E/S/P geometry ablation complete."
    ;;
  controls)
    run_mechanism_controls "$@"
    echo "Mechanism controls complete. Next run:"
    echo "  RUN_ROOT='$RUN_ROOT' DEVICE='$DEVICE' GPU_ID='$GPU_ID' bash '$SCRIPT_DIR/run_reproduction.sh' analysis"
    ;;
  analysis)
    if [[ $# -gt 0 ]]; then
      echo "The analysis stage does not accept training arguments: $*" >&2
      exit 2
    fi
    run_analysis
    ;;
  all)
    run_reference_ablation "$@"
    run_mechanism_controls "$@"
    run_analysis
    ;;
  *)
    echo "Unknown action '$ACTION'. Use: ablation, geometry, controls, analysis, or all." >&2
    exit 2
    ;;
esac

echo "Finished action '$ACTION'. Results: $RUN_ROOT"
