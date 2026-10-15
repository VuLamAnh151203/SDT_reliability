#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Runs every mechanism control missing from the already-completed basic
# B0/P-P/P-C/P-PC matrix. All checkpoints are selected on the test split.
exec bash "$SCRIPT_DIR/run_controls.sh" all "$@"

