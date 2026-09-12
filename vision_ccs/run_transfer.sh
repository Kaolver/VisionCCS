#!/bin/bash
#SBATCH --job-name=ccs-transfer
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Cross-category transfer: fit on one VQAv2 category, evaluate on another.
# A category-specific rule does not transfer; a truth direction should.
# Read the off-diagonal, and read CCS against CRC-TPC.
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

MODEL="${MODEL:-qwen2}"

if [ -d ./caches_v3 ]; then
  echo "=== v3 caches: answer position, mid-stack and final layer ==="
  for LAYER in 14 28; do
    echo ""
    echo "--- layer $LAYER ---"
    python transfer.py --cache-dir ./caches_v3 --cache-version v3 \
        --model "$MODEL" --layer "$LAYER" --position answer \
        --seeds 42 1 2 --selection val_consistency \
        --out "./transfer_${MODEL}_L${LAYER}.json"
  done
else
  echo "no ./caches_v3 -- falling back to the v1 final-layer caches"
  python transfer.py --cache-dir ./hidden_states_cache_final --cache-version v1 \
      --model "$MODEL" --seeds 42 1 2 --selection val_consistency \
      --out "./transfer_${MODEL}_v1.json"
fi

echo ""
echo "=== outputs ==="
ls -lh ./transfer_*.json
