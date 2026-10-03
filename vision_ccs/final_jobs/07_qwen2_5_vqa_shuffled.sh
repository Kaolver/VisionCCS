#!/bin/bash
#SBATCH --job-name=fj07_qwen2_5_vqa_shuffled
#SBATCH --output=final_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:15:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 07/07 -- final CCS analysis
#   model    : Qwen2.5-VL-7B (qwen2_5)
#   data     : VQAv2, all questions, SHUFFLED-IMAGE CONTROL (each question gets another image)
#   steps    : hidden states at every 4th layer -> layer sweep
#              (CCS, supervised probe, logistic regression with tuned C
#              -- all on the same grouped 60/40 test split, seeds 42 1)
#   expected : ~50 min, time limit 01:15:00
#   writes   : ./caches_final/vqa/hs_qwen2_5_*_shuffled.npz, ./final_results/sweep_qwen2_5_vqa_shuffled.json
#
# Re-submitting is safe: finished zero-shot runs, cached categories and
# finished sweep categories are kept, only the rest is redone.
# Submit from vision_ccs/:  bash final_jobs/submit_all.sh 07
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
mkdir -p ./caches_final/vqa ./final_results

echo "=== 2. hidden states (qwen2_5 / vqa_shuffled) ==="
MISSING=""
for CAT in object_detection attribute_recognition spatial_recognition; do
  [ -f ./caches_final/vqa/hs_qwen2_5_${CAT}_shuffled.npz ] || MISSING="$MISSING $CAT"
done
if [ -n "$MISSING" ]; then
  python extract.py --model qwen2_5 --vqa-json ./vqav2_mapped.json --categories $MISSING \
      --layer-stride 4 --out-dir ./caches_final/vqa --shuffle-images
else
  echo "  all categories cached"
fi
for CAT in object_detection attribute_recognition spatial_recognition; do
  if [ ! -f ./caches_final/vqa/hs_qwen2_5_${CAT}_shuffled.npz ]; then
    echo "ERROR: no hidden states for $CAT -- see the extract.py output above" >&2
    exit 1
  fi
done

echo "=== 3. layer sweep (qwen2_5 / vqa_shuffled) ==="
python layer_sweep.py --cache-dir ./caches_final/vqa --model qwen2_5 \
    --categories object_detection attribute_recognition spatial_recognition --shuffled \
    --positions final --seeds 42 1 --grouped --selection val_consistency \
    --pick-layer --resume \
    --out ./final_results/sweep_qwen2_5_vqa_shuffled.json
echo "=== done 07_qwen2_5_vqa_shuffled ==="
