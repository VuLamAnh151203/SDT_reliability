#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

FOLD_NAME="${FOLD_NAME:-checkpoint_split}"
MODELS="${MODELS:-B0 B1 B2 B3 B4 B5 B6}"
E3_SELECTION_SPLIT="${E3_SELECTION_SPLIT:-test}"
E3_SELECTION_METRIC="${E3_SELECTION_METRIC:-weighted_f1}"
REGENERATE_TEACHER="${REGENERATE_TEACHER:-0}"
RUN_E3_AUDIT="${RUN_E3_AUDIT:-1}"

TEACHER_FILE="$OUTPUT_DIR/manifests/train_impact_teacher.csv"
mkdir -p "$OUTPUT_DIR/manifests" "$OUTPUT_DIR/E3/checkpoints"

if [[ "$REGENERATE_TEACHER" == "1" || ! -s "$TEACHER_FILE" ]]; then
  echo "[E3 1/3] Generating train-only impact teacher..."
  "$PYTHON" "$SCRIPT_DIR/generate_train_impact_teacher.py" \
    "${MODEL_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --modalities text \
    --seed "$SEED" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[E3 1/3] Reusing impact teacher: $TEACHER_FILE"
fi

read -r -a SEED_ARRAY <<< "$SEEDS"
read -r -a MODEL_ARRAY <<< "$MODELS"

echo "[E3 2/3] Training models: ${MODEL_ARRAY[*]}"
echo "[E3 2/3] Seeds: ${SEED_ARRAY[*]}"
echo "[E3 2/3] Epoch-selection split: $E3_SELECTION_SPLIT"
echo "[E3 2/3] Epoch-selection metric: $E3_SELECTION_METRIC"
"$PYTHON" "$SCRIPT_DIR/e3_train_compare.py" \
  "${MODEL_ARGS[@]}" \
  "${ATTENTION_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --impact-teacher "$TEACHER_FILE" \
  --models "${MODEL_ARRAY[@]}" \
  --seeds "${SEED_ARRAY[@]}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --impact-eps "$IMPACT_EPS" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --max-batches "$MAX_BATCHES" \
  --fold-name "$FOLD_NAME" \
  --selection-split "$E3_SELECTION_SPLIT" \
  --selection-metric "$E3_SELECTION_METRIC"

if [[ "$RUN_E3_AUDIT" == "1" ]]; then
  TRAINED_CHECKPOINTS=()
  for seed in "${SEED_ARRAY[@]}"; do
    for model in "${MODEL_ARRAY[@]}"; do
      checkpoint="$OUTPUT_DIR/E3/checkpoints/${FOLD_NAME}_${model}_seed${seed}.pt"
      if [[ ! -s "$checkpoint" ]]; then
        echo "ERROR: missing E3 checkpoint: $checkpoint" >&2
        exit 2
      fi
      TRAINED_CHECKPOINTS+=("$checkpoint")
    done
  done

  echo "[E3 3/3] Running post-training influence audit..."
  "$PYTHON" "$SCRIPT_DIR/e3_posttraining_audit.py" \
    "${MODEL_ARGS[@]}" \
    --trained-checkpoints "${TRAINED_CHECKPOINTS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --split test \
    --modalities text \
    --strategies train_mean \
    --impact-eps "$IMPACT_EPS" \
    --max-conversations "$MAX_CONVERSATIONS"
else
  echo "[E3 3/3] Post-training audit skipped (RUN_E3_AUDIT=$RUN_E3_AUDIT)."
fi

echo "E3 complete. Results: $OUTPUT_DIR/E3"
