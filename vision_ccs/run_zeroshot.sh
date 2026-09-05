#!/bin/bash
#SBATCH --job-name=ccs-zeroshot
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# B1 zero-shot baseline, both prompt variants.
#
# The first run gave zero-shot an "Answer yes or no." task hint that CCS
# extraction never receives -- an asymmetry favouring zero-shot. Both variants
# are run so the size of that effect is measured rather than assumed:
#   (default)         with the hint      -> zeroshot_{model}_*.npz
#   --no-instruction  matched to CCS     -> zeroshot_{model}_noinstr_*.npz
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

for MODEL in qwen2 llava; do
  echo ""
  echo "=== $MODEL : with instruction (original) ==="
  python zero_shot.py --model "$MODEL" --out-dir ./zeroshot

  echo ""
  echo "=== $MODEL : no instruction (prompt-matched to CCS) ==="
  python zero_shot.py --model "$MODEL" --out-dir ./zeroshot       --no-instruction --tag _noinstr
done

echo ""
echo "=== outputs ==="
ls -lh ./zeroshot
