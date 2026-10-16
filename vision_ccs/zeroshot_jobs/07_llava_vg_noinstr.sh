#!/bin/bash
#SBATCH --job-name=zs07_llava_vg_noinstr
#SBATCH --output=zeroshot_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 07/12 -- zero-shot Yes/No baseline
#   model    : LLaVA-1.5-7B (llava)
#   dataset  : Visual Genome, 36,000 questions (object / attribute / spatial, 12,000 each)
#   prompt   : matched to CCS (no "Answer yes or no.")
#   expected : ~2-4 h (time limit 06:00:00)
#   writes   : zeroshot_report/zeroshot_llava_vg_noinstr_{object,attribute,spatial}.npz + _summary.json
#
# Submit from vision_ccs/:  bash zeroshot_jobs/submit_all.sh 07
# Capped rerun if too slow: LIMIT=2000 bash zeroshot_jobs/submit_all.sh 07
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1   # progress lines reach the Slurm log as they happen

python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json
python zero_shot.py --model llava \
    --vqa-json ./vg_mapped.json --categories object attribute spatial \
    --no-instruction --tag _vg_noinstr --out-dir ./zeroshot_report ${LIMIT:+--limit $LIMIT}
