#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

DATASET="${DATASET:-IEMOCAP}"
SEED="${SEED:-2024}"
GPU_ID="${GPU_ID:-0}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-test}"
VALID_RATIO="${VALID_RATIO:-0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/${DATASET,,}_${SELECTION_PROTOCOL}_seed${SEED}}"
RADIUS_MIN="${RADIUS_MIN:-0.20}"
RADIUS_INIT="${RADIUS_INIT:-0.75}"
RADIUS_MAX="${RADIUS_MAX:-0.90}"
LAMBDA_RADIUS="${LAMBDA_RADIUS:-0.01}"

case "$DATASET" in
  IEMOCAP)
    EPOCHS="${EPOCHS:-150}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    LR="${LR:-0.0001}"
    SDT_TEMPERATURE="${SDT_TEMPERATURE:-1}"
    # This experiment uses tical-mode=observe: anchor queries never affect
    # logits or either Wheel loss. Keep them disabled to avoid an unnecessary
    # O(utterances x anchor_size) search after epoch 10.
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-$EPOCHS}"
    ANCHOR_SIZE="${ANCHOR_SIZE:-2048}"
    ;;
  MELD)
    EPOCHS="${EPOCHS:-50}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.000005}"
    SDT_TEMPERATURE="${SDT_TEMPERATURE:-8}"
    WARMUP_EPOCHS="${WARMUP_EPOCHS:-$EPOCHS}"
    ANCHOR_SIZE="${ANCHOR_SIZE:-4096}"
    ;;
  *)
    echo "DATASET must be IEMOCAP or MELD; got: $DATASET" >&2
    exit 2
    ;;
esac

case "$SELECTION_PROTOCOL" in
  test|validation) ;;
  *)
    echo "SELECTION_PROTOCOL must be test or validation; got: $SELECTION_PROTOCOL" >&2
    exit 2
    ;;
esac

ALL_MODES=(fixed-shared learnable-shared learnable-per-class per-class-regularized)
if [[ "$MODE" == "all" ]]; then
  MODES=("${ALL_MODES[@]}")
else
  MODES=("$MODE")
fi

is_known_mode() {
  local candidate="$1"
  local known
  for known in "${ALL_MODES[@]}"; do
    [[ "$candidate" == "$known" ]] && return 0
  done
  return 1
}

run_one() {
  local radius_mode="$1"
  shift
  local lambda_radius=0
  if [[ "$radius_mode" == "per-class-regularized" ]]; then
    lambda_radius="$LAMBDA_RADIUS"
  fi

  echo "================================================================="
  echo "Prototype radius mode : $radius_mode"
  echo "Dataset / seed        : $DATASET / $SEED"
  echo "Checkpoint selection  : $SELECTION_PROTOCOL"
  echo "Radius min/init/max   : $RADIUS_MIN / $RADIUS_INIT / $RADIUS_MAX"
  echo "Radius regularization : $lambda_radius"
  echo "Output                 : $OUTPUT_DIR"
  echo "================================================================="

  "${PYTHON:-python}" -u "$ROOT_DIR/train.py" \
    --Dataset "$DATASET" --fusion-variant sdt \
    --fusion-feature-source enhanced \
    --use-tical --tical-mode observe --use-emotion-wheel \
    --wheel-geometry poincare --wheel-radius-mode free \
    --hyperbolic-dim 16 --hyp-eps 1e-5 --typicality-eps 1e-8 \
    --wheel-prototype-radius "$RADIUS_INIT" \
    --wheel-prototype-radius-min "$RADIUS_MIN" \
    --wheel-prototype-radius-max "$RADIUS_MAX" \
    --wheel-prototype-radius-mode "$radius_mode" \
    --wheel-temperature 1 --wheel-anchor-mix 0.5 \
    --lambda-wheel-proto 0.1 --lambda-wheel-cpcc 0.05 \
    --lambda-wheel-radius "$lambda_radius" \
    --wheel-semantic-control correct \
    --wheel-uncertainty-mode none --lambda-wheel-uncertainty 0 \
    --tical-warmup-epochs "$WARMUP_EPOCHS" \
    --anchor-size "$ANCHOR_SIZE" --anchor-conf-threshold 0.8 \
    --anchor-balance none --anchor-admission teacher \
    --anchor-min-per-class 0 \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --hidden-dim 1024 --n-head 8 \
    --lr "$LR" --l2 0.00001 --dropout 0.5 \
    --temp "$SDT_TEMPERATURE" \
    --gamma-1 1 --gamma-2 1 --gamma-3 1 \
    --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 --lambda-hyp 0 \
    --selection-protocol "$SELECTION_PROTOCOL" --valid-ratio "$VALID_RATIO" \
    --seed "$SEED" --device cuda --gpu-id "$GPU_ID" \
    --output-dir "$OUTPUT_DIR" "$@"
}

mkdir -p "$OUTPUT_DIR"
for radius_mode in "${MODES[@]}"; do
  if ! is_known_mode "$radius_mode"; then
    echo "Unknown mode '$radius_mode'. Use: ${ALL_MODES[*]} or all" >&2
    exit 2
  fi
  run_one "$radius_mode" "$@"
done

"${PYTHON:-python}" "$SCRIPT_DIR/summarize.py" "$OUTPUT_DIR"
echo "Finished. Summary: $OUTPUT_DIR/radius_ablation_runs.csv"
