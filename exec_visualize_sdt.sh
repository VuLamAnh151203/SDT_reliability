#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash exec_visualize_sdt.sh PATH/TO/best_checkpoint.pt [options]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/visualize_sdt_features.py" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  "$@"
