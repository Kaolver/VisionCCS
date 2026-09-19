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
source "$(dirname "$0")/_common.sh"

echo "=== running nonlinear_ccs.py ==="
PYTHONPATH="" python nonlinear_ccs.py
status=$?
echo "=== nonlinear_ccs.py finished at $(date) with exit code: $status ==="
exit $status
