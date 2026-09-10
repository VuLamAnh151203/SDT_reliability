#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train.py" \
  --Dataset IEMOCAP --fusion-variant guided \
  --epochs 150 --batch-size 16 --hidden-dim 1024 --n-head 8 \
  --lr 0.0001 --l2 0.00001 --dropout 0.5 --temp 1 \
  --gamma-1 1 --gamma-2 1 --gamma-3 1 \
  --lambda-co 0.1 --lambda-reg 0.0001 --seed 2024 "$@"
