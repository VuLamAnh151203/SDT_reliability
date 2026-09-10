#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/exec_iemocap.sh" \
  --fusion-variant guided \
  --distribution-init sdt-preserving \
  --initial-logvar -6 \
  --lambda-co 0.5 \
  --lambda-reg 0.01 \
  "$@"
