#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

EXPERIMENT="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
EXPERIMENT="${EXPERIMENT^^}"
DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/${DATASET,,}}"
PERMUTATION_SEED="${PERMUTATION_SEED:-1729}"

case "$DATASET" in
  IEMOCAP)
    EPOCHS="${EPOCHS:-150}"; BATCH_SIZE="${BATCH_SIZE:-16}"
    LR="${LR:-0.0001}"; TEMP="${TEMP:-1}"; ANCHOR_SIZE="${ANCHOR_SIZE:-2048}"
    ;;
  MELD)
    EPOCHS="${EPOCHS:-50}"; BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.000005}"; TEMP="${TEMP:-8}"; ANCHOR_SIZE="${ANCHOR_SIZE:-4096}"
    ;;
  *) echo "DATASET must be IEMOCAP or MELD" >&2; exit 2 ;;
esac

# These are the controls missing from the basic B0/P-P/P-C/P-PC ablation.
ALL_IDS=(P-0 RAND-P RAND-C RAND-PC DETACH-PC P-FIXED)
if [[ "$EXPERIMENT" == "ALL" ]]; then
  IDS=("${ALL_IDS[@]}")
else
  IDS=("$EXPERIMENT")
fi

run_one() {
  local id="$1" seed="$2"
  shift 2
  local proto=0 cpcc=0 semantic=correct radius_mode=free detach=0
  case "$id" in
    P-0) ;;
    RAND-P) proto=0.1; semantic=random ;;
    RAND-C) cpcc=0.05; semantic=random ;;
    RAND-PC) proto=0.1; cpcc=0.05; semantic=random ;;
    DETACH-PC) proto=0.1; cpcc=0.05; detach=1 ;;
    P-FIXED) proto=0.1; cpcc=0.05; radius_mode=fixed ;;
    *) echo "Unknown ID '$id'. Use: ${ALL_IDS[*]} or all" >&2; exit 2 ;;
  esac
  local detach_args=()
  [[ "$detach" == 1 ]] && detach_args=(--wheel-detach-features)

  echo "================================================================="
  echo "Wheel mechanism control=$id dataset=$DATASET seed=$seed"
  echo "proto=$proto cpcc=$cpcc semantic=$semantic radius=$radius_mode detach=$detach"
  echo "Checkpoint selection is locked to TEST."
  echo "================================================================="

  "${PYTHON:-python}" -u "$ROOT_DIR/train.py" \
    --Dataset "$DATASET" --fusion-variant sdt --fusion-feature-source enhanced \
    --use-tical --tical-mode observe --tical-warmup-epochs 10 \
    --anchor-size "$ANCHOR_SIZE" --anchor-conf-threshold 0.8 \
    --anchor-balance none --anchor-admission teacher --anchor-min-per-class 0 \
    --use-emotion-wheel --wheel-geometry poincare --hyperbolic-dim 16 \
    --wheel-radius-mode "$radius_mode" --wheel-fixed-radius 0.75 \
    --wheel-prototype-radius 0.75 --wheel-temperature 1.0 \
    --wheel-semantic-control "$semantic" \
    --wheel-permutation-seed "$PERMUTATION_SEED" \
    --lambda-wheel-proto "$proto" --lambda-wheel-cpcc "$cpcc" \
    --wheel-uncertainty-mode none --lambda-wheel-uncertainty 0 \
    --lambda-hyp 0 --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --hidden-dim 1024 --n-head 8 --lr "$LR" --l2 0.00001 \
    --dropout 0.5 --temp "$TEMP" --gamma-1 1 --gamma-2 1 --gamma-3 1 \
    --seed "$seed" --output-dir "$OUTPUT_DIR" \
    "$@" "${detach_args[@]}" --selection-protocol test
}

for id in "${IDS[@]}"; do
  for seed in $SEEDS; do
    run_one "$id" "$seed" "$@"
  done
done

echo "Finished controls in: $OUTPUT_DIR"

