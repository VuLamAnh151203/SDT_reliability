#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PATH="${OOF_TARGETS:-$SCRIPT_DIR/reliability_targets/iemocap_oof_seed2024.csv}"
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/build_oof_reliability.py" \
  --Dataset IEMOCAP \
  --folds 5 \
  --epochs 50 \
  --batch-size 16 \
  --hidden-dim 1024 \
  --n-head 8 \
  --dropout 0.5 \
  --lr 0.0001 \
  --l2 0.00001 \
  --temp 1 \
  --reliability-temperature 1 \
  --seed 2024 \
  --output "$TARGET_PATH" \
  "$@"
