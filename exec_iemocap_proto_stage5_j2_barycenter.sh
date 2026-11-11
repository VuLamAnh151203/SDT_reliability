#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 5: constrain each class's Poincare Frechet barycenter to shared q_c.
exec bash "$SCRIPT_DIR/exec_iemocap_proto_stage4_j2.sh" \
  --lambda-wheel-alignment "${LAMBDA_WHEEL_ALIGNMENT_BARYCENTER:-0.05}" \
  "$@"
