#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 4: two free prototypes per class/modality, no p-to-q alignment.
# Diversity prevents collapse; utilization discourages one permanently idle
# component. Both are small and can be overridden from the environment.
export LAMBDA_WHEEL_ALIGNMENT=0
exec bash "$SCRIPT_DIR/exec_iemocap_proto_stage3_loss_ramp.sh" \
  --wheel-prototypes-per-class 2 \
  --wheel-multi-prototype-temperature "${MULTIPROTO_TEMPERATURE:-0.20}" \
  --wheel-multi-prototype-init-std "${MULTIPROTO_INIT_STD:-0.02}" \
  --wheel-prototype-diversity-margin "${MULTIPROTO_DIVERSITY_MARGIN:-0.50}" \
  --lambda-wheel-diversity "${LAMBDA_WHEEL_DIVERSITY:-0.01}" \
  --lambda-wheel-utilization "${LAMBDA_WHEEL_UTILIZATION:-0.01}" \
  "$@"
