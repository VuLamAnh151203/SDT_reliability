#!/usr/bin/env bash
# Shared configuration for the four independent entry points.
# Prefer environment overrides, for example:
#   CHECKPOINT=/path/to/best_checkpoint.pt DEVICE=cuda bash run_task_a_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SDT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

: "${CHECKPOINT:?Set CHECKPOINT to a fully trained SDT_new best_checkpoint.pt}"

PYTHON="${PYTHON:-python}"
FEATURE_PATH="${FEATURE_PATH:-$SDT_ROOT/../SDT/data/iemocap_multimodal_features.pkl}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-auto}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2024}"
NUM_THREADS="${NUM_THREADS:-0}"
MAX_CONVERSATIONS="${MAX_CONVERSATIONS:-0}"
ALLOW_SMOKE_CHECKPOINT="${ALLOW_SMOKE_CHECKPOINT:-0}"

checkpoint_name="$(basename -- "$CHECKPOINT")"
checkpoint_name="${checkpoint_name%.*}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/outputs/${checkpoint_name}_${SPLIT}}"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --feature-path "$FEATURE_PATH"
  --run-dir "$RUN_DIR"
  --split "$SPLIT"
  --device "$DEVICE"
  --gpu-id "$GPU_ID"
  --seed "$SEED"
  --num-threads "$NUM_THREADS"
  --max-conversations "$MAX_CONVERSATIONS"
)

if [[ "$ALLOW_SMOKE_CHECKPOINT" == "1" ]]; then
  COMMON_ARGS+=(--allow-smoke-checkpoint)
fi

