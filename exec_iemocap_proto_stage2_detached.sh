#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 2: stage 1 with Wheel gradients detached from SDT features.
exec bash "$SCRIPT_DIR/exec_iemocap_proto_stage1_modality_d32.sh" \
  --wheel-detach-features \
  "$@"
