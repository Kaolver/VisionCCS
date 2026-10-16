#!/bin/bash
#SBATCH --job-name=fj05_qwen2_vg
#SBATCH --output=final_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:15:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 05/07 -- final CCS analysis
#   model    : Qwen2-VL-7B (qwen2)
#   data     : Visual Genome, 2000 questions per category
#   steps    : zero-shot if missing -> hidden states at every 4th layer -> layer sweep
#              (CCS, supervised probe, logistic regression with tuned C, zero-shot
#              -- all on the same grouped 60/40 test split, seeds 42 1 2)
#   expected : ~50 min, time limit 01:15:00
#   writes   : ./caches_final/vg/hs_qwen2_*.npz, ./final_results/sweep_qwen2_vg.json
#
# Re-submitting is safe: finished zero-shot runs, cached categories and
# finished sweep categories are kept, only the rest is redone.
# Submit from vision_ccs/:  bash final_jobs/submit_all.sh 05
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
mkdir -p ./caches_final/vg ./final_results
python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json

echo "=== 1. zero-shot (qwen2 / vg), only what is missing ==="
for V in noinstr instr; do
  if [ "$V" = noinstr ]; then TAG=_vg_noinstr; FLAG=--no-instruction; else TAG=_vg; FLAG=; fi
  if [ -f ./zeroshot_report/zeroshot_qwen2${TAG}_summary.json ]; then
    echo "  have zeroshot_qwen2${TAG}"
  else
    python zero_shot.py --model qwen2 --vqa-json ./vg_mapped.json --categories object attribute spatial \
        $FLAG --tag "$TAG" --out-dir ./zeroshot_report
  fi
done

echo "=== 2. hidden states (qwen2 / vg) ==="
MISSING=""
for CAT in object attribute spatial; do
  [ -f ./caches_final/vg/hs_qwen2_${CAT}.npz ] || MISSING="$MISSING $CAT"
done
if [ -n "$MISSING" ]; then
  python extract.py --model qwen2 --vqa-json ./vg_mapped.json --categories $MISSING \
      --layer-stride 4 --out-dir ./caches_final/vg --limit 2000
else
  echo "  all categories cached"
fi
for CAT in object attribute spatial; do
  if [ ! -f ./caches_final/vg/hs_qwen2_${CAT}.npz ]; then
    echo "ERROR: no hidden states for $CAT -- see the extract.py output above" >&2
    exit 1
  fi
done

echo "=== 3. layer sweep (qwen2 / vg) ==="
python layer_sweep.py --cache-dir ./caches_final/vg --model qwen2 \
    --categories object attribute spatial \
    --positions final --seeds 42 1 2 --grouped --selection val_consistency \
    --pick-layer --resume \
    --zeroshot-dir ./zeroshot_report --zeroshot-tags _vg_noinstr _vg \
    --out ./final_results/sweep_qwen2_vg.json
echo "=== done 05_qwen2_vg ==="
