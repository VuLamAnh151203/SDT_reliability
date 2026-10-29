#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="${1:-$SCRIPT_DIR/results/train}"
if [[ $# -gt 0 ]]; then shift; fi
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/analysis}"

exec "${PYTHON:-python}" -u "$SCRIPT_DIR/analyze.py" \
  "$TRAIN_DIR" --output-dir "$OUTPUT_DIR" "$@"

