#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONTROL_DIR="${1:-$SCRIPT_DIR/results/iemocap}"
REFERENCE_DIR="${2:-$ROOT_DIR/geometry_analysis/results/iemocap}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -d "$CONTROL_DIR" ]]; then
  echo "Control result directory does not exist: $CONTROL_DIR" >&2
  exit 2
fi

mapfile -t CONTROL_CHECKPOINTS < <(find "$CONTROL_DIR" -type f -name best_checkpoint.pt | sort)
if [[ ${#CONTROL_CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No control best_checkpoint.pt found under: $CONTROL_DIR" >&2
  echo "Available mechanism result directories containing checkpoints:" >&2
  find "$SCRIPT_DIR/results" -type f -name best_checkpoint.pt -printf '%h\n' 2>/dev/null \
    | sort -u >&2 || true
  echo "Run run_all.sh first, or pass the actual control result directory." >&2
  exit 2
fi

echo "Found ${#CONTROL_CHECKPOINTS[@]} control checkpoints under: $CONTROL_DIR"

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

