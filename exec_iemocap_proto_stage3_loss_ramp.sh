#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 3: detached features plus a delayed, linear prototype-loss schedule.
WARMUP="${PROTOTYPE_LOSS_WARMUP_EPOCHS:-15}"
RAMP="${PROTOTYPE_LOSS_RAMP_EPOCHS:-10}"
exec bash "$SCRIPT_DIR/exec_iemocap_proto_stage2_detached.sh" \
  --prototype-loss-warmup-epochs "$WARMUP" \
  --prototype-loss-ramp-epochs "$RAMP" \
  "$@"
