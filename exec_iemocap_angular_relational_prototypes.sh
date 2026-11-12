#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Full proposed pipeline:
#   p_c = Exp_0(U q_c + alpha r_c^perp),  U^T U = I,
#   r_c^perp = (I - U U^T) r_c,
# trained by shared-modality Prototype CE, angular relational KL, and the
# orthogonal-residual anchor penalty.  The original feature CPCC and prototype
# fusion are disabled so this run isolates the method described in the paper.
export TICAL_MODE=observe
exec bash "$SCRIPT_DIR/exec_iemocap_tical.sh" \
  --use-emotion-wheel \
  --wheel-geometry poincare \
  --wheel-feature-source pure \
  --hyperbolic-dim "${HYPERBOLIC_DIM:-16}" \
  --wheel-prototype-layout angular-residual \
  --wheel-prototype-radius-mode fixed-shared \
  --wheel-prototype-radius "${ANGULAR_REFERENCE_BALL_RADIUS:-0.75}" \
  --wheel-angular-residual-scale "${ANGULAR_RESIDUAL_SCALE:-1.0}" \
  --wheel-angular-residual-init-std "${ANGULAR_RESIDUAL_INIT_STD:-0.02}" \
  --wheel-prototype-structure-mode relational-kl \
  --wheel-relation-target-temperature "${RELATION_TARGET_TEMPERATURE:-0.5}" \
  --wheel-relation-prototype-temperature "${RELATION_PROTOTYPE_TEMPERATURE:-0.5}" \
  --wheel-temperature "${PROTOTYPE_TEMPERATURE:-1.0}" \
  --lambda-wheel-proto "${LAMBDA_PROTOTYPE_CE:-0.1}" \
  --lambda-wheel-structure "${LAMBDA_RELATION:-0.05}" \
  --lambda-wheel-anchor "${LAMBDA_ANCHOR:-0.01}" \
  --lambda-wheel-cpcc 0 \
  --lambda-wheel-radius 0 \
  --lambda-wheel-boundary 0 \
  --lambda-wheel-alignment 0 \
  --lambda-wheel-private-residual 0 \
  --lambda-wheel-diversity 0 \
  --lambda-wheel-utilization 0 \
  --lambda-hyp 0 \
  --prototype-evidence-fusion none \
  --prototype-evidence-weight 0 \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  "$@"
