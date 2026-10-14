#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$EXPERIMENT_DIR/configs/experiment_config.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Same B4 hyperparameters as the previous ~77.13 validation run, but using the
# requested legacy/exploratory protocol: select the best epoch on test.
E04_OUTPUT_DIR="${E04_OUTPUT_DIR:-$OUTPUT_DIR/Experiment_04_test_selection}"
E04_SEEDS="${E04_SEEDS:-2026}"
E04_EPOCHS="${E04_EPOCHS:-30}"
E04_PATIENCE="${E04_PATIENCE:-7}"
E04_GATE_LR="${E04_GATE_LR:-0.0003}"
E04_INITIAL_GATE="${E04_INITIAL_GATE:-0.90}"
E04_LAMBDA_IDENTITY="${E04_LAMBDA_IDENTITY:-0.001}"
E04_LAMBDA_IMPACT="${E04_LAMBDA_IMPACT:-0.10}"
E04_SELECTION_METRIC="${E04_SELECTION_METRIC:-weighted_f1}"
E04_REUSE_COMPLETED_RUN="${E04_REUSE_COMPLETED_RUN:-1}"

echo "Experiment 04: B4 with test-set checkpoint selection"
echo "Seeds: $E04_SEEDS"
echo "initial_gate=$E04_INITIAL_GATE"
echo "lambda_impact=$E04_LAMBDA_IMPACT"
echo "gate_lr=$E04_GATE_LR"
echo "gate_modalities=text audio visual"
echo "selection=test/$E04_SELECTION_METRIC"
echo "WARNING: test-set selection makes the reported test score optimistically biased."

if [[ "$E04_REUSE_COMPLETED_RUN" == "1" \
      && -s "$E04_OUTPUT_DIR/E3/manifest.json" ]]; then
  echo "Reusing completed Experiment 04: $E04_OUTPUT_DIR/E3"
else
  DIAGNOSTIC_OUTPUT_DIR="$E04_OUTPUT_DIR" \
  DIAGNOSTIC_MODELS="B4" \
  DIAGNOSTIC_SEEDS="$E04_SEEDS" \
  DIAGNOSTIC_EPOCHS="$E04_EPOCHS" \
  DIAGNOSTIC_PATIENCE="$E04_PATIENCE" \
  DIAGNOSTIC_GATE_LR="$E04_GATE_LR" \
  DIAGNOSTIC_SELECTION_SPLIT="test" \
  DIAGNOSTIC_SELECTION_METRIC="$E04_SELECTION_METRIC" \
  DIAGNOSTIC_INITIAL_GATE="$E04_INITIAL_GATE" \
  DIAGNOSTIC_GATE_MODALITIES="text audio visual" \
  DIAGNOSTIC_LAMBDA_IDENTITY="$E04_LAMBDA_IDENTITY" \
  DIAGNOSTIC_LAMBDA_IMPACT="$E04_LAMBDA_IMPACT" \
  DIAGNOSTIC_IMPACT_MIN_MARGIN="0" \
  DIAGNOSTIC_DEFER_TEST_EVALUATION="0" \
  bash "$SCRIPT_DIR/run_e3_gate_only_diagnostic.sh"
fi

echo "Experiment 04 complete: $E04_OUTPUT_DIR/E3"
echo "Primary result: $E04_OUTPUT_DIR/E3/model_comparison.csv"
