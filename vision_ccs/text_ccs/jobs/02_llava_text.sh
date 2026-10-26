#!/bin/bash
#SBATCH --job-name=tc02_llava_text
#SBATCH --output=text_ccs/jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# LLaVA-1.5-7B without images, raw and chat
MODEL=llava
FORMATS="raw chat"
source "${SLURM_SUBMIT_DIR:-$HOME/VisionCCS/vision_ccs}/text_ccs/jobs/_text_job.sh"
