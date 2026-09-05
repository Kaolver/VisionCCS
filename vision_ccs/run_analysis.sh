#!/bin/bash
#SBATCH --job-name=ccs-analysis
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Re-runnable end-to-end analysis on the existing caches:
# headline table -> selection criteria -> item-level zero-shot cross-tab.
# Everything here is batch; nothing is meant to be run on a login node.

source "$(dirname "$0")/_slurm_common.sh"

OUT=./reanalysis_final.json

echo "=== 1. headline re-analysis (val_consistency selection) ==="
python reanalysis.py --cache-dir ./hidden_states_cache_final --models qwen2 \
    --seeds 42 1 2 3 4 --selection val_consistency --out "$OUT"

echo ""
echo "=== 2. label-free selection criteria ==="
python select_criteria.py "$OUT" --combine

echo ""
echo "=== 3. item-level CCS vs zero-shot ==="
if [ -d ./zeroshot ]; then
  python compare_zeroshot.py "$OUT" --zeroshot-dir ./zeroshot
else
  echo "no ./zeroshot -- run run_zeroshot.sh first"
fi
