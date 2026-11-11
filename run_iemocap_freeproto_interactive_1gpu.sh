#!/usr/bin/env bash
set -euo pipefail

# Run the free-prototype Emotion Wheel experiment on exactly one GPU inside an
# existing VinUni CECS Slurm interactive session.
#
# Start the allocation before running this file, for example:
#   srun --qos=interactive --gres=gpu:1 --cpus-per-task=4 \
#     --mem=16G --time=04:00:00 --pty bash
#
# If the current allocation exposes two GPUs, select the first or second one:
#   GPU_SLOT=0 bash run_iemocap_freeproto_interactive_1gpu.sh
#   GPU_SLOT=1 bash run_iemocap_freeproto_interactive_1gpu.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Error: this runner must be launched inside a Slurm interactive session." >&2
  echo "Start one with:" >&2
  echo "  srun --qos=interactive --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash" >&2
  exit 2
fi

GPU_SLOT="${GPU_SLOT:-0}"
if [[ ! "$GPU_SLOT" =~ ^[0-9]+$ ]]; then
  echo "Error: GPU_SLOT must be a zero-based integer; got '$GPU_SLOT'." >&2
  exit 2
fi

# Slurm may expose physical indexes (for example 2,4) or GPU UUIDs. Select by
# position in its existing list instead of assuming that physical GPU 0 or 1
# belongs to this job.
ALLOCATED_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$ALLOCATED_DEVICES" ]]; then
  IFS=',' read -r -a DEVICE_LIST <<< "$ALLOCATED_DEVICES"
  if (( GPU_SLOT >= ${#DEVICE_LIST[@]} )); then
    echo "Error: GPU_SLOT=$GPU_SLOT, but Slurm exposed only ${#DEVICE_LIST[@]} GPU(s): $ALLOCATED_DEVICES" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="${DEVICE_LIST[$GPU_SLOT]}"
else
  # This fallback is useful on installations where the device cgroup is
  # enforced but CUDA_VISIBLE_DEVICES is not populated.
  export CUDA_VISIBLE_DEVICES="$GPU_SLOT"
fi

# Resolve a Python that actually contains Torch. An explicit PYTHON always
# wins. The final fallback is the migrated mmrec environment currently used on
# research-gateway; using the executable directly is sufficient and does not
# modify or activate the parent shell.
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$HOME/venvs/ml/bin/python" ]]; then
  PYTHON_BIN="$HOME/venvs/ml/bin/python"
elif [[ -x "$HOME/_container_layer_20261108/miniconda3/envs/mmrec/bin/python3.12" ]]; then
  PYTHON_BIN="$HOME/_container_layer_20261108/miniconda3/envs/mmrec/bin/python3.12"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: no Python executable was found. Set PYTHON=/path/to/python." >&2
  exit 2
fi

ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-4}"
# One GPU gets at most four of the eight CPUs in the current two-GPU
# allocation. Override CPU_THREADS explicitly if a future allocation differs.
if (( ALLOCATED_CPUS < 4 )); then
  DEFAULT_CPU_THREADS="$ALLOCATED_CPUS"
else
  DEFAULT_CPU_THREADS=4
fi
CPU_THREADS="${CPU_THREADS:-$DEFAULT_CPU_THREADS}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/freeproto_1gpu}"
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-exec_iemocap_wheel_free_prototypes.sh}"
if [[ "$EXPERIMENT_SCRIPT" == */* || ! -f "$SCRIPT_DIR/$EXPERIMENT_SCRIPT" ]]; then
  echo "Error: invalid experiment runner '$EXPERIMENT_SCRIPT'." >&2
  exit 2
fi

echo "Slurm job:       $SLURM_JOB_ID"
echo "Allocated GPUs:  ${ALLOCATED_DEVICES:-<not set>}"
echo "Selected GPU:    $CUDA_VISIBLE_DEVICES (visible to PyTorch as cuda:0)"
echo "CPU threads:     $CPU_THREADS"
echo "Data workers:    $NUM_WORKERS"
echo "Python:          $PYTHON_BIN"
echo "Output:          $OUTPUT_DIR"
echo "Experiment:      $EXPERIMENT_SCRIPT"

export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside this interactive session")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        "Expected exactly one visible GPU, found {}".format(
            torch.cuda.device_count()))
print("PyTorch GPU:     {}".format(torch.cuda.get_device_name(0)))
print("CUDA version:    {}".format(torch.version.cuda))
sys.stdout.flush()
PY

export PYTHON="$PYTHON_BIN"
exec bash "$SCRIPT_DIR/$EXPERIMENT_SCRIPT" \
  --device cuda \
  --gpu-id 0 \
  --num-threads "$CPU_THREADS" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory \
  --output-dir "$OUTPUT_DIR" \
  "$@"
