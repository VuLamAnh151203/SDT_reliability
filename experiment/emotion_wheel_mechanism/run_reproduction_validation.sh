#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Validation protocol:
#   - split VALID_RATIO from the original training conversations;
#   - select best_checkpoint.pt using validation weighted F1;
#   - evaluate the selected checkpoint once on the unchanged test split.
export SELECTION_PROTOCOL=validation
export VALID_RATIO="${VALID_RATIO:-0.1}"
export RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/results/reproduction_validation}"

exec bash "$SCRIPT_DIR/run_reproduction.sh" "$@"
