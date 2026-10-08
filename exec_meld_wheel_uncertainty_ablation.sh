#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "$EXPERIMENT" == "all" ]]; then
  MODES=(none gate supervise full)
else
  MODES=("$EXPERIMENT")
fi

for mode in "${MODES[@]}"; do
  echo "============================================================"
  echo "MELD wheel uncertainty ablation: $mode"
  echo "============================================================"
  bash "$SCRIPT_DIR/exec_meld_wheel_uncertainty.sh" "$mode" "$@"
done
