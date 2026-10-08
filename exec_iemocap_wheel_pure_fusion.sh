#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Controlled ablation:
#   SDT encoders + Poincare Wheel Prototype/CPCC losses
#   gate, student classifiers and final fusion use H_TT/H_AA/H_VV directly.
# Uncertainty, CA-KD, HypCPCC, TiCAL fusion adjustment and COLD remain disabled.
exec bash "$SCRIPT_DIR/exec_iemocap_wheel_uncertainty.sh" none \
  --fusion-feature-source pure \
  "$@"
