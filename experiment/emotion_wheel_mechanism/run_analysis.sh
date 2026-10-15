#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONTROL_DIR="${1:-$SCRIPT_DIR/results/iemocap}"
REFERENCE_DIR="${2:-$ROOT_DIR/geometry_analysis/results/iemocap}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"

echo "Summarizing controls plus existing B0/P-P/P-C/P-PC references..."
"${PYTHON:-python}" -u "$SCRIPT_DIR/summarize_mechanism.py" \
  "$CONTROL_DIR" "$REFERENCE_DIR" \
  --output-dir "$CONTROL_DIR/mechanism_summary"

echo "Running geometry diagnostics for every new control checkpoint..."
bash "$ROOT_DIR/geometry_analysis/run_diagnostics.sh" \
  "$CONTROL_DIR" --device "$DEVICE" --gpu-id "$GPU_ID"

echo "Running pure/enhanced representation probes for controls..."
"${PYTHON:-python}" -u "$SCRIPT_DIR/probe_representations.py" \
  "$CONTROL_DIR" --device "$DEVICE" --gpu-id "$GPU_ID" \
  --output "$CONTROL_DIR/mechanism_summary/control_representation_probes.csv"

echo "Analysis complete: $CONTROL_DIR/mechanism_summary"
echo "For the existing TRUE-PC checkpoint, run gradient_alignment.py using the command in README.md."

