#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

EXPERIMENT="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXPERIMENT="${EXPERIMENT^^}"
DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/${DATASET,,}}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-validation}"
VALID_RATIO="${VALID_RATIO:-0.1}"
WHEEL_DIM="${WHEEL_DIM:-16}"
WHEEL_RADIUS="${WHEEL_RADIUS:-0.75}"
WHEEL_TEMPERATURE="${WHEEL_TEMPERATURE:-1.0}"

case "$DATASET" in
  IEMOCAP)
    EPOCHS="${EPOCHS:-150}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    LR="${LR:-0.0001}"
    SDT_TEMPERATURE="${SDT_TEMPERATURE:-1}"
    ANCHOR_SIZE="${ANCHOR_SIZE:-2048}"
    ;;
  MELD)
    EPOCHS="${EPOCHS:-50}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.000005}"
    SDT_TEMPERATURE="${SDT_TEMPERATURE:-8}"
    ANCHOR_SIZE="${ANCHOR_SIZE:-4096}"
    ;;
  *)
    echo "DATASET must be IEMOCAP or MELD; got: $DATASET" >&2
    exit 2
    ;;
esac

ALL_IDS=(B0 E-P E-C E-PC S-P S-C S-PC P-P P-C P-PC)
if [[ "$EXPERIMENT" == "ALL" ]]; then
  IDS=("${ALL_IDS[@]}")
else
  IDS=("$EXPERIMENT")
fi

is_known_id() {
  local candidate="$1"
  local known
  for known in "${ALL_IDS[@]}"; do
    [[ "$candidate" == "$known" ]] && return 0
  done
  return 1
}

run_one() {
  local experiment_id="$1"
  local seed="$2"
  shift 2
  local geometry=""
  local lambda_proto="0"
  local lambda_cpcc="0"
  local variant_args=()

  case "$experiment_id" in
    B0) ;;
    E-P)  geometry="euclidean"; lambda_proto="0.1" ;;
    E-C)  geometry="euclidean"; lambda_cpcc="0.05" ;;
    E-PC) geometry="euclidean"; lambda_proto="0.1"; lambda_cpcc="0.05" ;;
    S-P)  geometry="spherical"; lambda_proto="0.1" ;;
    S-C)  geometry="spherical"; lambda_cpcc="0.05" ;;
    S-PC) geometry="spherical"; lambda_proto="0.1"; lambda_cpcc="0.05" ;;
    P-P)  geometry="poincare"; lambda_proto="0.1" ;;
    P-C)  geometry="poincare"; lambda_cpcc="0.05" ;;
    P-PC) geometry="poincare"; lambda_proto="0.1"; lambda_cpcc="0.05" ;;
  esac

  if [[ "$experiment_id" != "B0" ]]; then
    # observe keeps original SDT fusion/KD and isolates the two Wheel losses.
    variant_args=(
      --use-tical --tical-mode observe --use-emotion-wheel
      --wheel-geometry "$geometry"
      --hyperbolic-dim "$WHEEL_DIM"
      --wheel-prototype-radius "$WHEEL_RADIUS"
      --wheel-temperature "$WHEEL_TEMPERATURE"
      --lambda-wheel-proto "$lambda_proto"
      --lambda-wheel-cpcc "$lambda_cpcc"
    )
  fi

  echo "================================================================="
  echo "Experiment=$experiment_id dataset=$DATASET seed=$seed"
  echo "geometry=${geometry:-none} proto=$lambda_proto cpcc=$lambda_cpcc"
  echo "output=$OUTPUT_DIR"
  echo "================================================================="

  "${PYTHON:-python}" -u "$ROOT_DIR/train.py" \
    --Dataset "$DATASET" --fusion-variant sdt \
    --selection-protocol "$SELECTION_PROTOCOL" --valid-ratio "$VALID_RATIO" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --hidden-dim 1024 --n-head 8 \
    --lr "$LR" --l2 0.00001 --dropout 0.5 --temp "$SDT_TEMPERATURE" \
    --gamma-1 1 --gamma-2 1 --gamma-3 1 \
    --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 --lambda-hyp 0 \
    --tical-warmup-epochs 5 --anchor-size "$ANCHOR_SIZE" \
    --anchor-conf-threshold 0.8 --anchor-balance none \
    --anchor-admission teacher --anchor-min-per-class 0 \
    --seed "$seed" --output-dir "$OUTPUT_DIR" \
    "${variant_args[@]}" "$@"
}

for id in "${IDS[@]}"; do
  if ! is_known_id "$id"; then
    echo "Unknown experiment '$id'. Use: ${ALL_IDS[*]} or all" >&2
    exit 2
  fi
  for seed in $SEEDS; do
    run_one "$id" "$seed" "$@"
  done
done

echo "All requested runs finished. Aggregate with:"
echo "  ${PYTHON:-python} $SCRIPT_DIR/summarize.py $OUTPUT_DIR"
