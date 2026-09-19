#!/bin/bash
#SBATCH --job-name=linear-ccs
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --time=03:00:00
#SBATCH --mem=32G

# Stage 1: linear CCS probe. Populates ./hidden_states_cache_final, which every later stage reuses (use_cache=True).
#
# Runs on a MIG slice (1/7 A100, ~10GB). If a job dies with CUDA OOM, lower
# 'batch_size' in linear_ccs.py CONFIG (now 8) rather than raising --mem:
# --mem is host RAM and does not change the GPU slice.
source "$(dirname "$0")/_common.sh"

echo "=== running linear_ccs.py ==="
PYTHONPATH="" python linear_ccs.py
status=$?
echo "=== linear_ccs.py finished at $(date) with exit code: $status ==="
exit $status
