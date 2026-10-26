#!/bin/bash
#SBATCH --job-name=tc08_vision_reeval
#SBATCH --output=text_ccs/jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# burns_ccs.py on the final_jobs caches in ./caches_final/{vqa,vg}
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
R=./text_ccs_results
mkdir -p "$R"
for D in vqa vg; do
  for M in llava qwen2 qwen2_5; do
    if ! ls ./caches_final/$D/hs_${M}_*.npz >/dev/null 2>&1; then
      echo "  no ./caches_final/$D caches for $M, skipped"; continue
    fi
    echo "=== $M / $D ==="
    python text_ccs/evaluate_text.py --model $M --vision-cache-dir ./caches_final/$D \
        --vision-dataset $D --positions final --last-layer-positions answer \
        --seeds 0 1 2 --resume --out "$R/${M}_vision-${D}.json"
  done
done
python text_ccs/summarize.py --results-dir "$R" --out-dir ./text_ccs_report
echo "=== done 08_vision_reeval ==="
