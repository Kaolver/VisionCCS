#!/bin/bash
#SBATCH --job-name=fj02_qwen2_vqa
#SBATCH --output=final_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:45:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 02/07 -- final CCS analysis
#   model    : Qwen2-VL-7B (qwen2)
#   data     : VQAv2, all questions
#   steps    : zero-shot if missing -> hidden states at every 4th layer -> layer sweep
#              (CCS, supervised probe, logistic regression with tuned C, zero-shot
#              -- all on the same grouped 60/40 test split, seeds 42 1 2)
#   expected : ~1.25 h (also runs the missing CCS-prompt zero-shot), time limit 01:45:00
#   writes   : ./caches_final/vqa/hs_qwen2_*.npz, ./final_results/sweep_qwen2_vqa.json
#
# Re-submitting is safe: finished zero-shot runs, cached categories and
# finished sweep categories are kept, only the rest is redone.
# Submit from vision_ccs/:  bash final_jobs/submit_all.sh 02
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
mkdir -p ./caches_final/vqa ./final_results

echo "=== 1. zero-shot (qwen2 / vqa), only what is missing ==="
for V in noinstr instr; do
  if [ "$V" = noinstr ]; then TAG=_vqa_noinstr; FLAG=--no-instruction; else TAG=_vqa; FLAG=; fi
  if [ -f ./zeroshot_report/zeroshot_qwen2${TAG}_summary.json ]; then
    echo "  have zeroshot_qwen2${TAG}"
  else
    python zero_shot.py --model qwen2 --vqa-json ./vqav2_mapped.json --categories object_detection attribute_recognition spatial_recognition \
        $FLAG --tag "$TAG" --out-dir ./zeroshot_report
  fi
done

echo "=== 2. hidden states (qwen2 / vqa) ==="
MISSING=""
for CAT in object_detection attribute_recognition spatial_recognition; do
  [ -f ./caches_final/vqa/hs_qwen2_${CAT}.npz ] || MISSING="$MISSING $CAT"
done
if [ -n "$MISSING" ]; then
  python extract.py --model qwen2 --vqa-json ./vqav2_mapped.json --categories $MISSING \
      --layer-stride 4 --out-dir ./caches_final/vqa
else
  echo "  all categories cached"
fi
for CAT in object_detection attribute_recognition spatial_recognition; do
  if [ ! -f ./caches_final/vqa/hs_qwen2_${CAT}.npz ]; then
    echo "ERROR: no hidden states for $CAT -- see the extract.py output above" >&2
    exit 1
  fi
done

echo "=== 3. layer sweep (qwen2 / vqa) ==="
python layer_sweep.py --cache-dir ./caches_final/vqa --model qwen2 \
    --categories object_detection attribute_recognition spatial_recognition \
    --positions final --seeds 42 1 2 --grouped --selection val_consistency \
    --pick-layer --resume \
    --zeroshot-dir ./zeroshot_report --zeroshot-tags _vqa_noinstr _vqa \
    --out ./final_results/sweep_qwen2_vqa.json
echo "=== done 02_qwen2_vqa ==="
