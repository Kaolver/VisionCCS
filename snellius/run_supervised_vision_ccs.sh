#!/bin/bash
#SBATCH --job-name=snellius-vision-ccs
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=32G

# Load modules that provide Python and CUDA
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1 # Load CUDA for PyTorch to use

cd /gpfs/home6/mdemirev/snellius/

source venv/bin/activate

echo "=== Running supervised_vision_ccs.py ==="
# Set PYTHONPATH to empty to avoid conflicts
PYTHONPATH="" python supervised_vision_ccs.py

echo "=== Job finished at $(date) with exit code: $? ==="
