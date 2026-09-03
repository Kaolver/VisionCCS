#!/bin/bash
#SBATCH --job-name=ccs-extract-v3
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Phase 1: all-layer extraction at corrected token positions, plus the
# shuffled-image control, for qwen2 and llava.

cd "$HOME/VisionCCS/vision_ccs" || exit 1
source venv_ccs/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for MODEL in qwen2 llava; do
  echo "=== $MODEL : real images ==="
  python extract.py --model "$MODEL" --layer-stride 2 --out-dir ./caches_v3

  echo "=== $MODEL : shuffled-image control ==="
  python extract.py --model "$MODEL" --layer-stride 2 --out-dir ./caches_v3 \
      --shuffle-images
done

echo "=== cache sizes ==="
du -sh ./caches_v3
ls -lh ./caches_v3
