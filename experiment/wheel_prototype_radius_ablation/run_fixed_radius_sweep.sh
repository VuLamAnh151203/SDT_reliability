#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
GPU_ID="${GPU_ID:-0}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/fixed_radius_sweep_${DATASET,,}_${SELECTION_PROTOCOL}}"

# A Poincare-ball point must satisfy ||p|| < 1.  The requested 1.0 control is
# therefore represented by a point very close to the boundary.
RADIUS_ONE_EFFECTIVE="${RADIUS_ONE_EFFECTIVE:-0.9999}"
RADIUS_MIN="${RADIUS_MIN:-0.05}"
RADIUS_MAX="${RADIUS_MAX:-0.99998}"

REQUESTED_RADII=(0.25 0.50 0.75 1.00)
EFFECTIVE_RADII=(0.25 0.50 0.75 "$RADIUS_ONE_EFFECTIVE")
RADIUS_LABELS=(0p25 0p50 0p75 1p00)

mkdir -p "$OUTPUT_DIR"

read -r -a seed_values <<< "$SEEDS"
for seed in "${seed_values[@]}"; do
  for index in "${!REQUESTED_RADII[@]}"; do
    requested="${REQUESTED_RADII[$index]}"
    effective="${EFFECTIVE_RADII[$index]}"
    label="${RADIUS_LABELS[$index]}"
    run_output="$OUTPUT_DIR/radius_${label}/seed_${seed}"

    echo "================================================================="
    echo "Fixed prototype-radius sweep"
    echo "Dataset / seed             : $DATASET / $seed"
    echo "Requested prototype radius : $requested"
    echo "Effective prototype radius : $effective"
    echo "Output                     : $run_output"
    echo "================================================================="

    DATASET="$DATASET" \
    SEED="$seed" \
    GPU_ID="$GPU_ID" \
    SELECTION_PROTOCOL="$SELECTION_PROTOCOL" \
    OUTPUT_DIR="$run_output" \
    RADIUS_MIN="$RADIUS_MIN" \
    RADIUS_INIT="$effective" \
    RADIUS_MAX="$RADIUS_MAX" \
      bash "$SCRIPT_DIR/run_ablation.sh" fixed-shared "$@"
  done
done

"${PYTHON:-python}" "$SCRIPT_DIR/summarize_fixed_radius_sweep.py" \
  "$OUTPUT_DIR"

echo "Finished fixed-radius sweep."
echo "CSV    : $OUTPUT_DIR/fixed_radius_sweep_runs.csv"
echo "Report : $OUTPUT_DIR/fixed_radius_sweep_summary.md"

