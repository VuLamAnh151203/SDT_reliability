#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  echo "Error: set INIT_CHECKPOINT to stage 5 best_prototype_checkpoint.pt." >&2
  exit 2
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Error: checkpoint does not exist: $INIT_CHECKPOINT" >&2
  exit 2
fi

# Stage 6: freeze the trained SDT/prototype branch and train only calibrated
# evidence parameters plus the supervised SDT-vs-prototype alpha selector.
export WHEEL_HYPERBOLIC_DIM=32
export LAMBDA_WHEEL_ALIGNMENT=0
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_modality_prototypes.sh" \
  --wheel-prototypes-per-class 2 \
  --wheel-detach-features \
  --wheel-multi-prototype-temperature "${MULTIPROTO_TEMPERATURE:-0.20}" \
  --wheel-multi-prototype-init-std "${MULTIPROTO_INIT_STD:-0.02}" \
  --wheel-prototype-diversity-margin "${MULTIPROTO_DIVERSITY_MARGIN:-0.50}" \
  --prototype-loss-warmup-epochs 0 \
  --prototype-loss-ramp-epochs 0 \
  --lambda-wheel-proto 0 \
  --lambda-wheel-structure 0 \
  --lambda-wheel-alignment 0 \
  --lambda-wheel-boundary 0 \
  --lambda-wheel-diversity 0 \
  --lambda-wheel-utilization 0 \
  --prototype-evidence-fusion calibrated \
  --prototype-evidence-weight "${PROTOTYPE_EVIDENCE_WEIGHT:-0.20}" \
  --prototype-evidence-hidden-dim "${PROTOTYPE_EVIDENCE_HIDDEN_DIM:-32}" \
  --prototype-evidence-warmup-epochs 0 \
  --prototype-evidence-ramp-epochs 0 \
  --prototype-evidence-initial-alpha "${PROTOTYPE_EVIDENCE_INITIAL_ALPHA:-0.05}" \
  --lambda-prototype-selector "${LAMBDA_PROTOTYPE_SELECTOR:-1.0}" \
  --prototype-selector-target-temperature "${PROTOTYPE_SELECTOR_TARGET_TEMPERATURE:-1.0}" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --freeze-for-selector \
  "$@"
