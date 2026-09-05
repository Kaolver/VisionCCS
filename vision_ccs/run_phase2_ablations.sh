#!/bin/bash
#SBATCH --job-name=ccs-phase2-ablation
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Phase 2.1 + 2.2 on the EXISTING caches -- no re-extraction needed.
#
# 2.1 --weight-norm unit : constrain ||w||=1, removing the scale degeneracy that
#     lets the CCS loss reach ~0 along any separating direction. Prediction is
#     NOT higher accuracy; it is that corr(loss, accuracy) turns strongly
#     negative, i.e. the loss becomes a usable selection criterion again.
# 2.2 --no-var-normalize : the CCS reference defaults to var_normalize=False and
#     we have been running True throughout. Never tested.
#
# 2x2 grid. logreg is skipped: it is sklearn, cannot take the constraint, and is
# the slowest part of a run.

source "$(dirname "$0")/_slurm_common.sh"

CACHE=./hidden_states_cache_final
COMMON="--cache-dir $CACHE --models qwen2 --seeds 42 1 2 3 4 \
        --selection val_consistency --skip-logreg"

for WN in none unit; do
  for VN in "" "--no-var-normalize"; do
    TAG="wn-${WN}$( [ -n "$VN" ] && echo "_varnorm-off" || echo "_varnorm-on" )"
    echo ""
    echo "######################################################################"
    echo "# weight_norm=${WN}  var_normalize=$( [ -n "$VN" ] && echo off || echo on )"
    echo "######################################################################"
    python reanalysis.py $COMMON --weight-norm "$WN" $VN \
        --out "./ablation_${TAG}.json"

    echo "--- selection-criterion comparison for ${TAG} ---"
    python select_criteria.py "./ablation_${TAG}.json" --combine
  done
done

echo ""
echo "=== outputs ==="
ls -lh ./ablation_*.json
