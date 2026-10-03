#!/bin/bash
#SBATCH --job-name=zs05_qwen2_5_vqa_noinstr
#SBATCH --output=zeroshot_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 05/12 -- zero-shot Yes/No baseline
#   model    : Qwen2.5-VL-7B (qwen2_5)
#   dataset  : VQAv2, 5,648 questions (object 1,306 / attribute 3,366 / spatial 976)
#   prompt   : matched to CCS (no "Answer yes or no.")
#   expected : ~30-45 min (time limit 01:30:00)
#   writes   : zeroshot_report/zeroshot_qwen2_5_vqa_noinstr_{object_detection,attribute_recognition,spatial_recognition}.npz + _summary.json
#
# Submit from vision_ccs/:  bash zeroshot_jobs/submit_all.sh 05
# Capped rerun if too slow: LIMIT=2000 bash zeroshot_jobs/submit_all.sh 05
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e

python zero_shot.py --model qwen2_5 \
    --vqa-json ./vqav2_mapped.json --categories object_detection attribute_recognition spatial_recognition \
    --no-instruction --tag _vqa_noinstr --out-dir ./zeroshot_report ${LIMIT:+--limit $LIMIT}
