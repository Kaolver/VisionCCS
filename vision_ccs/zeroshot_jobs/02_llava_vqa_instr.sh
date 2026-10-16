#!/bin/bash
#SBATCH --job-name=zs02_llava_vqa_instr
#SBATCH --output=zeroshot_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 02/12 -- zero-shot Yes/No baseline
#   model    : LLaVA-1.5-7B (llava)
#   dataset  : VQAv2, 5,648 questions (object 1,306 / attribute 3,366 / spatial 976)
#   prompt   : with "Answer yes or no." instruction
#   expected : ~30-45 min (time limit 01:30:00)
#   writes   : zeroshot_report/zeroshot_llava_vqa_{object_detection,attribute_recognition,spatial_recognition}.npz + _summary.json
#
# Submit from vision_ccs/:  bash zeroshot_jobs/submit_all.sh 02
# Capped rerun if too slow: LIMIT=2000 bash zeroshot_jobs/submit_all.sh 02
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1   # progress lines reach the Slurm log as they happen

python zero_shot.py --model llava \
    --vqa-json ./vqav2_mapped.json --categories object_detection attribute_recognition spatial_recognition \
    --tag _vqa --out-dir ./zeroshot_report ${LIMIT:+--limit $LIMIT}
