#!/bin/bash
#SBATCH --job-name=tc07_qwen2_5_llm_raw
#SBATCH --output=text_ccs/jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Qwen2.5-7B-Instruct, raw
MODEL=qwen2_5_llm
FORMATS="raw"
source "${SLURM_SUBMIT_DIR:-$HOME/VisionCCS/vision_ccs}/text_ccs/jobs/_text_job.sh"
