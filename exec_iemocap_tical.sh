#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# SDT + TiCAL only.  This deliberately selects the deterministic SDT branch
# and sets every COLD/OOF-reliability weight to zero.
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train.py" \
  --Dataset IEMOCAP --fusion-variant sdt --use-tical \
  --tical-mode "${TICAL_MODE:-kd}" \
  --tical-warmup-epochs 5 \
  --anchor-size 2048 --anchor-conf-threshold 0.8 \
  --hyperbolic-dim 128 --hyp-eps 1e-5 --typicality-eps 1e-8 \
  --consistency-t 0.2 --consistency-k 0.5 \
  --beta-gate 1.0 --lambda-hyp 0.1 \
  --epochs 150 --batch-size 16 --hidden-dim 1024 --n-head 8 \
  --lr 0.0001 --l2 0.00001 --dropout 0.5 --temp 1 \
  --gamma-1 1 --gamma-2 1 --gamma-3 1 \
  --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 \
  --modality-prune-quantile 0 --sample-prune-quantile 0 \
  --seed 2024 "$@"
