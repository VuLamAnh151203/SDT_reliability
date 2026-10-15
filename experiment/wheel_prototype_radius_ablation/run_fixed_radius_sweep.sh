#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-IEMOCAP}"
SEEDS="${SEEDS:-2024}"
GPU_ID="${GPU_ID:-0}"
SELECTION_PROTOCOL="${SELECTION_PROTOCOL:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/fixed_radius_sweep_${DATASET,,}_${SELECTION_PROTOCOL}}"

# Match the TRUE-PC reference run that produced 75.306 weighted F1.  These
# values intentionally override the speed-oriented defaults in
# run_ablation.sh.  DataLoader workers can advance PyTorch's loader generator
# differently and thereby change mini-batch order even when --seed is fixed.
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NUM_THREADS="${NUM_THREADS:-0}"

# A Poincare-ball point must satisfy ||p|| < 1.  The requested 1.0 control is
# therefore represented by a point very close to the boundary.
RADIUS_ONE_EFFECTIVE="${RADIUS_ONE_EFFECTIVE:-0.9999}"
RADIUS_MIN="${RADIUS_MIN:-0.05}"
RADIUS_MAX="${RADIUS_MAX:-0.99998}"
RADII="${RADII:-0.25 0.50 0.75 1.00}"

mkdir -p "$OUTPUT_DIR"

read -r -a seed_values <<< "$SEEDS"
read -r -a radius_values <<< "$RADII"
for seed in "${seed_values[@]}"; do
  for requested in "${radius_values[@]}"; do
    case "$requested" in
      0.25|.25) requested="0.25"; effective="0.25"; label="0p25" ;;
      0.50|0.5|.5) requested="0.50"; effective="0.50"; label="0p50" ;;
      0.75|.75) requested="0.75"; effective="0.75"; label="0p75" ;;
      1.00|1.0|1) requested="1.00"; effective="$RADIUS_ONE_EFFECTIVE"; label="1p00" ;;
      *)
        echo "Unknown radius '$requested'. Use: 0.25 0.50 0.75 1.00" >&2
        exit 2
        ;;
    esac
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
    WARMUP_EPOCHS="$WARMUP_EPOCHS" \
    NUM_WORKERS="$NUM_WORKERS" \
    NUM_THREADS="$NUM_THREADS" \
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

