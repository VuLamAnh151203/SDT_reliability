#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Compatibility launcher: run the complete uncertainty-aware model once.
# All supplied arguments are forwarded to train.py through the main launcher.
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_uncertainty.sh" full "$@"
