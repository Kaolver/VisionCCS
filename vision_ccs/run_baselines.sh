#!/bin/bash
#SBATCH --job-name=ccs-baselines
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# CCS vs the unsupervised baselines it has to clear. CRC-TPC is Burns' own
# second method; in text, PCA on contrast-pair differences reaches ~97% of CCS.
#
# Also sweeps the normalization axis, which was never tested:
#   per_split    Burns-faithful, transductive
#   train_stats  inductive
#   cluster      cluster-norm (2407.18712), inductive
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

CACHE="${CACHE:-./hidden_states_cache_final}"
MODEL="${MODEL:-qwen2}"

for NORM in per_split train_stats cluster; do
  echo ""
  echo "######################################################################"
  echo "# normalization = $NORM"
  echo "######################################################################"
  python reanalysis.py --cache-dir "$CACHE" --models "$MODEL" \
      --seeds 42 1 2 3 4 --selection val_consistency --skip-logreg \
      --norm "$NORM" --controls \
      --out "./baselines_${MODEL}_${NORM}.json"
done

echo ""
echo "=== outputs ==="
ls -lh ./baselines_*.json
