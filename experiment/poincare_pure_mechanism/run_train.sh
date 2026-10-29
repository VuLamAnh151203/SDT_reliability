#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

EXPERIMENT="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
EXPERIMENT="${EXPERIMENT^^}"

SEEDS="${SEEDS:-2024}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/train}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-test}"
VALID_RATIO="${VALID_RATIO:-0.1}"
EPOCHS="${EPOCHS:-150}"
WHEEL_TEMPERATURE="${WHEEL_TEMPERATURE:-1.0}"

# P-0 is the no-loss projector control.  E-E, B-E and B-P isolate map versus
# metric. S-S is the equal-dimensional hyperspherical comparison.
ALL_IDS=(B0 P-0 E-E B-E S-S B-P)

known_id() {
  local wanted="$1" item
  for item in "${ALL_IDS[@]}"; do
    [[ "$wanted" == "$item" ]] && return 0
  done
  return 1
}

run_one() {
  local id="$1" seed="$2"
  shift 2
  local geometry="poincare" proto="0.1" cpcc="0.05"
  local wheel_args=()

  case "$id" in
    B0)  proto="0"; cpcc="0" ;;
    P-0) proto="0"; cpcc="0" ;;
    E-E) geometry="euclidean" ;;
    B-E) geometry="ball_euclidean" ;;
    S-S) geometry="spherical" ;;
    B-P) geometry="poincare" ;;
    *) echo "Unknown ID: $id" >&2; exit 2 ;;
  esac

  if [[ "$id" != "B0" ]]; then
    wheel_args=(
      --use-tical --tical-mode observe --use-emotion-wheel
      --wheel-feature-source pure --wheel-geometry "$geometry"
      --hyperbolic-dim 16 --wheel-prototype-radius 0.75
      --wheel-temperature "$WHEEL_TEMPERATURE"
      --lambda-wheel-proto "$proto" --lambda-wheel-cpcc "$cpcc"
    )
  fi

  echo "================================================================="
  echo "ID=$id seed=$seed geometry=$geometry proto=$proto cpcc=$cpcc"
  echo "selection=$SELECTION_PROTOCOL output=$OUTPUT_DIR"
  echo "================================================================="

  "${PYTHON:-python}" -u "$ROOT_DIR/train.py" \
    --Dataset IEMOCAP --fusion-variant sdt \
    --selection-protocol "$SELECTION_PROTOCOL" --valid-ratio "$VALID_RATIO" \
    --epochs "$EPOCHS" --batch-size 16 --hidden-dim 1024 --n-head 8 \
    --lr 0.0001 --l2 0.00001 --dropout 0.5 --temp 1 \
    --gamma-1 1 --gamma-2 1 --gamma-3 1 \
    --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 --lambda-hyp 0 \
    --tical-warmup-epochs 10 --anchor-size 2048 \
    --anchor-conf-threshold 0.8 --anchor-balance none \
    --anchor-admission teacher --anchor-min-per-class 0 \
    --num-workers 0 --num-threads 0 \
    --seed "$seed" --output-dir "$OUTPUT_DIR" \
    "${wheel_args[@]}" "$@"
}

if [[ "$EXPERIMENT" == "ALL" ]]; then
  IDS=("${ALL_IDS[@]}")
else
  known_id "$EXPERIMENT" || {
    echo "Use one of: ${ALL_IDS[*]} or all" >&2
    exit 2
  }
  IDS=("$EXPERIMENT")
fi

for id in "${IDS[@]}"; do
  for seed in $SEEDS; do
    run_one "$id" "$seed" "$@"
  done
done

echo "Training finished: $OUTPUT_DIR"

