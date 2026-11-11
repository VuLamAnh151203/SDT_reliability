#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# No fixed prototype participates in this experiment.  The original
# Prototype CE (0.10) is redirected to three independently learned T/A/V
# prototype banks.  Every [C,D] bank starts at random full-D directions with
# a safe Poincare radius and is optimized from epoch one.  Feature-level Wheel
# CPCC remains unchanged; prototype-level Wheel CPCC constrains each bank.
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_only.sh" \
  --use-data-prototypes \
  --data-prototype-layout modality-specific \
  --data-prototype-init random \
  --data-prototype-initial-radius "${DATA_PROTOTYPE_INITIAL_RADIUS:-0.3}" \
  --replace-fixed-wheel-prototypes \
  --data-prototype-warmup-epochs 0 \
  --data-prototype-ramp-epochs 0 \
  --lambda-wheel-proto "${LAMBDA_LEARNED_PROTOTYPE:-0.10}" \
  --lambda-wheel-cpcc "${LAMBDA_FEATURE_CPCC:-0.05}" \
  --lambda-data-prototype 0 \
  --lambda-data-prototype-cpcc "${LAMBDA_PROTOTYPE_CPCC:-0.01}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  "$@"
