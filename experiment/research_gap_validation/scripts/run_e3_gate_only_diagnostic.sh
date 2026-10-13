#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

DIAGNOSTIC_OUTPUT_DIR="${DIAGNOSTIC_OUTPUT_DIR:-$OUTPUT_DIR/diagnostics/gate_only}"
DIAGNOSTIC_MODELS="${DIAGNOSTIC_MODELS:-B0 B1 B2 B3 B4 B6}"
DIAGNOSTIC_SEEDS="${DIAGNOSTIC_SEEDS:-2024}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-30}"
DIAGNOSTIC_PATIENCE="${DIAGNOSTIC_PATIENCE:-7}"
DIAGNOSTIC_GATE_LR="${DIAGNOSTIC_GATE_LR:-0.0003}"
DIAGNOSTIC_SELECTION_SPLIT="${DIAGNOSTIC_SELECTION_SPLIT:-test}"
DIAGNOSTIC_SELECTION_METRIC="${DIAGNOSTIC_SELECTION_METRIC:-weighted_f1}"

TEACHER_FILE="$OUTPUT_DIR/manifests/train_impact_teacher.csv"
mkdir -p "$DIAGNOSTIC_OUTPUT_DIR" "$OUTPUT_DIR/manifests"

if [[ ! -s "$TEACHER_FILE" ]]; then
  echo "[diagnostic 1/2] Generating train-only impact teacher..."
  "$PYTHON" "$SCRIPT_DIR/generate_train_impact_teacher.py" \
    "${MODEL_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --modalities text \
    --seed "$SEED" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[diagnostic 1/2] Reusing impact teacher: $TEACHER_FILE"
fi

read -r -a MODEL_ARRAY <<< "$DIAGNOSTIC_MODELS"
read -r -a SEED_ARRAY <<< "$DIAGNOSTIC_SEEDS"

echo "[diagnostic 2/2] Frozen SDT/classifier; training gates only"
echo "Models: ${MODEL_ARRAY[*]}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Gate LR: $DIAGNOSTIC_GATE_LR"

"$PYTHON" "$SCRIPT_DIR/e3_train_compare.py" \
  "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --output-dir "$DIAGNOSTIC_OUTPUT_DIR" \
  --impact-teacher "$TEACHER_FILE" \
  --models "${MODEL_ARRAY[@]}" \
  --seeds "${SEED_ARRAY[@]}" \
  --epochs "$DIAGNOSTIC_EPOCHS" \
  --patience "$DIAGNOSTIC_PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --lr "$DIAGNOSTIC_GATE_LR" \
  --gate-lr "$DIAGNOSTIC_GATE_LR" \
  --impact-eps "$IMPACT_EPS" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --max-batches "$MAX_BATCHES" \
  --fold-name gate_only_diagnostic \
  --selection-split "$DIAGNOSTIC_SELECTION_SPLIT" \
  --selection-metric "$DIAGNOSTIC_SELECTION_METRIC" \
  --freeze-classifier

echo "Gate-only diagnostic complete: $DIAGNOSTIC_OUTPUT_DIR/E3"
