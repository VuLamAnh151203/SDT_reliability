#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Common MELD configuration for SDT/TiCAL ablations.
#
# The default run is deterministic SDT without COLD or TiCAL.  Enable one
# TiCAL component by appending, for example:
#   --use-tical --tical-mode kd       # CA-KD
#   --use-tical --tical-mode hyp      # CA-KD + HypCPCC
#   --use-tical --tical-mode fusion   # CA-KD + fusion adjustment
#   --use-tical --tical-mode full     # all TiCAL components
#
# Anchor-quality ablations are available through:
#   bash exec_meld_anchor_experiment.sh combined --gpu-id 0
#
# Arguments appended by the caller occur last and therefore override these
# common defaults.
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/train.py" \
  --Dataset MELD --fusion-variant sdt \
  --tical-warmup-epochs 5 \
  --anchor-size 4096 --anchor-conf-threshold 0.8 \
  --anchor-balance none --anchor-admission teacher --anchor-min-per-class 0 \
  --hyperbolic-dim 16 --hyp-eps 1e-5 --typicality-eps 1e-8 \
  --consistency-t 0.2 --consistency-k 0.5 \
  --beta-gate 1.0 --lambda-hyp 0.1 \
  --lambda-wheel-proto 0 --lambda-wheel-cpcc 0 \
  --epochs 50 --batch-size 8 --hidden-dim 1024 --n-head 8 \
  --lr 0.000005 --l2 0.00001 --dropout 0.5 --temp 8 \
  --gamma-1 1 --gamma-2 1 --gamma-3 1 \
  --lambda-co 0 --lambda-reg 0 --lambda-reliability 0 \
  --seed 2024 "$@"
