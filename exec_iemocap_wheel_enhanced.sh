#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# SDT + Poincare Emotion Wheel on enhanced representations H'_T/H'_A/H'_V.
# This keeps the original enhanced SDT fusion and enables only the two Wheel
# objectives: prototype CE (0.1) and semantic CPCC (0.05).
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_only.sh" \
  --wheel-feature-source enhanced \
  "$@"
