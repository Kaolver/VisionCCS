#!/bin/bash
#SBATCH --job-name=ccs-extract-v3
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Phase 1: all-layer extraction at corrected token positions, plus the
# shuffled-image control, for qwen2 and llava.
#
# TEMPLATES=all adds the multi-template caches (5x the forward passes). Burns
# runs 8-13 templates per dataset; with one there is no surface variation for
# truth to be consistent across. TEMPLATES=plain reproduces the existing caches.
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

TEMPLATES="${TEMPLATES:-plain}"
MODELS="${MODELS:-qwen2 llava}"

for MODEL in $MODELS; do
  echo "=== $MODEL : real images (templates: $TEMPLATES) ==="
  python extract.py --model "$MODEL" --layer-stride 2 --out-dir ./caches_v3 \
      --templates $TEMPLATES

  echo "=== $MODEL : shuffled-image control ==="
  python extract.py --model "$MODEL" --layer-stride 2 --out-dir ./caches_v3 \
      --templates $TEMPLATES --shuffle-images
done

echo "=== cache sizes ==="
du -sh ./caches_v3
ls -lh ./caches_v3

# B3: banner-distractor caches -- the only control that tests identifiability
# rather than performance.
for MODEL in $MODELS; do
  echo "=== $MODEL : banner-distractor control ==="
  python extract.py --model "$MODEL" --layer-stride 2 --out-dir ./caches_v3 \
      --templates $TEMPLATES --distractor banner
done
