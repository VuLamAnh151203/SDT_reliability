#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Full SDT + TiCAL with fixed emotion-wheel prototypes in a 2-D Poincare ball.
# Every option can still be overridden by appending it to this command.
export TICAL_MODE="${TICAL_MODE:-full}"
exec bash "$SCRIPT_DIR/exec_iemocap_tical.sh" \
  --use-emotion-wheel \
  --hyperbolic-dim 2 \
  --wheel-prototype-radius 0.75 \
  --wheel-temperature 1.0 \
  --wheel-anchor-mix 0.5 \
  --lambda-wheel-proto 0.1 \
  --lambda-wheel-cpcc 0.05 \
  "$@"
