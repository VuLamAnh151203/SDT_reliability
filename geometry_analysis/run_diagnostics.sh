#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash geometry_analysis/run_diagnostics.sh CHECKPOINT_OR_RESULTS_DIR [options]" >&2
  exit 2
fi

TARGET="$1"
shift
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/analyze_geometry.py" \
  "$TARGET" --split test "$@"
