#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep the complete Emotion-Wheel objective on q_c and p_c^m, then make the
# modality-specific hyperbolic prototype evidence participate in final logits.
FUSION_MODE="${PROTOTYPE_EVIDENCE_FUSION:-class-wise}"
FUSION_WEIGHT="${PROTOTYPE_EVIDENCE_WEIGHT:-0.20}"
FUSION_HIDDEN="${PROTOTYPE_EVIDENCE_HIDDEN_DIM:-32}"
FUSION_WARMUP="${PROTOTYPE_EVIDENCE_WARMUP_EPOCHS:-5}"
FUSION_RAMP="${PROTOTYPE_EVIDENCE_RAMP_EPOCHS:-5}"
export WHEEL_HYPERBOLIC_DIM="${WHEEL_HYPERBOLIC_DIM:-32}"

exec bash "$SCRIPT_DIR/exec_iemocap_wheel_modality_prototypes.sh" \
  --prototype-evidence-fusion "$FUSION_MODE" \
  --prototype-evidence-weight "$FUSION_WEIGHT" \
  --prototype-evidence-hidden-dim "$FUSION_HIDDEN" \
  --prototype-evidence-warmup-epochs "$FUSION_WARMUP" \
  --prototype-evidence-ramp-epochs "$FUSION_RAMP" \
  "$@"
