#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${PROTOTYPE_MODALITY_WEIGHT_MODE:-uniform}"
WEIGHTS="${PROTOTYPE_MODALITY_WEIGHTS:-1 1 1}"
read -r -a WEIGHT_ARRAY <<< "$WEIGHTS"
if (( ${#WEIGHT_ARRAY[@]} != 3 )); then
  echo "Error: PROTOTYPE_MODALITY_WEIGHTS must contain exactly T A V." >&2
  exit 2
fi

# Keep the strongest current angular pipeline fixed and change only the
# weighting of T/A/V Prototype CE. Feature CPCC remains uniformly averaged.
exec bash "$SCRIPT_DIR/exec_iemocap_angular_relational_prototypes.sh" \
  --lambda-wheel-cpcc "${LAMBDA_WHEEL_CPCC:-0.05}" \
  --prototype-modality-weight-mode "$MODE" \
  --prototype-modality-weights "${WEIGHT_ARRAY[@]}" \
  --prototype-modality-weight-warmup-epochs "${PROTOTYPE_WEIGHT_WARMUP_EPOCHS:-5}" \
  --prototype-modality-weight-ramp-epochs "${PROTOTYPE_WEIGHT_RAMP_EPOCHS:-10}" \
  --prototype-modality-weight-ema-momentum "${PROTOTYPE_WEIGHT_EMA_MOMENTUM:-0.9}" \
  --prototype-modality-weight-temperature "${PROTOTYPE_WEIGHT_TEMPERATURE:-1.0}" \
  --prototype-modality-weight-minimum "${PROTOTYPE_WEIGHT_MINIMUM:-0.15}" \
  "$@"
