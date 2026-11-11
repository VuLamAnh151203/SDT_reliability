#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the original fixed 2-D Wheel bank and both losses that produced
# 75.31 weighted F1. Add one independent, shared D=16 prototype bank:
#   - initialized from class centroids during warm-up,
#   - optimized by supervised prototype CE,
#   - softly constrained by prototype-level Wheel CPCC.
# Environment variables expose the new ablation without changing the baseline.
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_only.sh" \
  --wheel-prototype-layout fixed-wheel \
  --use-data-prototypes \
  --lambda-data-prototype "${LAMBDA_DATA_PROTOTYPE:-0.02}" \
  --lambda-data-prototype-cpcc "${LAMBDA_DATA_PROTOTYPE_CPCC:-0.01}" \
  --data-prototype-warmup-epochs "${DATA_PROTOTYPE_WARMUP_EPOCHS:-5}" \
  --data-prototype-ramp-epochs "${DATA_PROTOTYPE_RAMP_EPOCHS:-10}" \
  --data-prototype-temperature "${DATA_PROTOTYPE_TEMPERATURE:-1.0}" \
  --grad-clip "${GRAD_CLIP:-0}" \
  "$@"
