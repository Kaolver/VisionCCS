#!/bin/bash
#SBATCH --job-name=nonlinear-ccs
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --time=03:00:00
#SBATCH --mem=32G

# Stage 3: non-linear (MLP) CCS probe. Reuses the stage-1 cache.
#
# Runs on a MIG slice (1/7 A100, ~10GB). If a job dies with CUDA OOM, lower
# 'batch_size' in linear_ccs.py CONFIG (now 8) rather than raising --mem:
# --mem is host RAM and does not change the GPU slice.
# NOTE: SLURM copies this script to a spool dir, so $(dirname "$0") is NOT the
# submit dir. Resolve _common.sh explicitly.
for D in "$SLURM_SUBMIT_DIR" "$HOME/VisionCCS/vision_ccs" "$(dirname "$0")" .; do
    if [ -n "$D" ] && [ -f "$D/_common.sh" ]; then
        source "$D/_common.sh"
        found_common=1
        break
    fi
done
if [ -z "$found_common" ]; then
    echo "ERROR: _common.sh not found (looked in SLURM_SUBMIT_DIR, ~/VisionCCS/vision_ccs)" >&2
    exit 1
fi

echo "=== running nonlinear_ccs.py ==="
PYTHONPATH="" python nonlinear_ccs.py
status=$?
echo "=== nonlinear_ccs.py finished at $(date) with exit code: $status ==="
exit $status
