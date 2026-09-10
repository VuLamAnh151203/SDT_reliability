#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PATH="${OOF_TARGETS:-$SCRIPT_DIR/reliability_targets/iemocap_oof_seed2024.csv}"
if [[ ! -f "$TARGET_PATH" ]]; then
  echo "OOF target file not found: $TARGET_PATH" >&2
  echo "Run exec_iemocap_build_oof_reliability.sh first or set OOF_TARGETS." >&2
  exit 1
fi
exec bash "$SCRIPT_DIR/exec_iemocap.sh" \
  --fusion-variant oof-guided \
  --oof-reliability-targets "$TARGET_PATH" \
  --lambda-reliability 1.0 \
  --modality-prune-quantile 0.10 \
  --sample-prune-quantile 0.05 \
  --disagreement-weight 1.0 \
  "$@"
