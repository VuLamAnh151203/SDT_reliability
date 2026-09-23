#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Anchor ablations for MELD. CA-KD + HypCPCC is used by default; append a
# later --tical-mode argument or set TICAL_MODE to select another TiCAL loss.
#
# Usage:
#   bash exec_meld_anchor_experiment.sh equal
#   bash exec_meld_anchor_experiment.sh coverage
#   bash exec_meld_anchor_experiment.sh modality
#   bash exec_meld_anchor_experiment.sh combined --gpu-id 0

EXPERIMENT="combined"
if [[ $# -gt 0 && "$1" != --* ]]; then
  EXPERIMENT="$1"
  shift
fi

case "$EXPERIMENT" in
  equal)
    EXTRA_ARGS=(
      --anchor-balance equal
      --anchor-admission teacher
      --anchor-min-per-class 0
    )
    ;;
  coverage)
    EXTRA_ARGS=(
      --anchor-balance equal
      --anchor-admission teacher
      --anchor-min-per-class "${ANCHOR_MIN_PER_CLASS:-8}"
    )
    ;;
  modality)
    EXTRA_ARGS=(
      --anchor-balance equal
      --anchor-admission modality
      --anchor-min-per-class 0
    )
    ;;
  combined)
    EXTRA_ARGS=(
      --anchor-balance equal
      --anchor-admission modality
      --anchor-min-per-class "${ANCHOR_MIN_PER_CLASS:-8}"
    )
    ;;
  *)
    echo "Unknown experiment '$EXPERIMENT'. Use: equal, coverage, modality, or combined." >&2
    exit 2
    ;;
esac

exec bash "$SCRIPT_DIR/exec_meld.sh" \
  --use-tical \
  --tical-mode "${TICAL_MODE:-hyp}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
