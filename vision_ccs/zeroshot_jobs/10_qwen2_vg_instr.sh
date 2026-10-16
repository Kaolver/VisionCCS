#!/bin/bash
#SBATCH --job-name=zs10_qwen2_vg_instr
#SBATCH --output=zeroshot_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 10/12 -- zero-shot Yes/No baseline
#   model    : Qwen2-VL-7B (qwen2)
#   dataset  : Visual Genome, 36,000 questions (object / attribute / spatial, 12,000 each)
#   prompt   : with "Answer yes or no." instruction
#   expected : ~2-4 h (time limit 06:00:00)
#   writes   : zeroshot_report/zeroshot_qwen2_vg_{object,attribute,spatial}.npz + _summary.json
#
# Submit from vision_ccs/:  bash zeroshot_jobs/submit_all.sh 10
# Capped rerun if too slow: LIMIT=2000 bash zeroshot_jobs/submit_all.sh 10
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1   # progress lines reach the Slurm log as they happen

python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json
python zero_shot.py --model qwen2 \
    --vqa-json ./vg_mapped.json --categories object attribute spatial \
    --tag _vg --out-dir ./zeroshot_report ${LIMIT:+--limit $LIMIT}
