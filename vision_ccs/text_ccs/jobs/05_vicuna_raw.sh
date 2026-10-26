#!/bin/bash
#SBATCH --job-name=tc05_vicuna_raw
#SBATCH --output=text_ccs/jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Vicuna-7B-v1.5, raw
MODEL=vicuna
FORMATS="raw"
source "${SLURM_SUBMIT_DIR:-$HOME/VisionCCS/vision_ccs}/text_ccs/jobs/_text_job.sh"
