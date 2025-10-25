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

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Create and activate virtual environment (if not exists)
if [ ! -d "venv" ]; then
    echo "=== Creating new Python 3.11 venv... ==="
    python -m venv venv
fi

source venv/bin/activate

echo "=== Installing/upgrading dependencies in venv ==="
pip install --upgrade pip
# Install PyTorch 2.5.1 and Torchvision (latest stable for CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# Install transformers (latest version supports Qwen2-VL)
pip install transformers accelerate pillow tqdm scikit-learn "numpy<2" qwen-vl-utils

echo "=== Checking if vision_ccs.py exists ==="
ls -lh vision_ccs.py

echo "=== Running vision_ccs.py ==="
# Set PYTHONPATH to empty to avoid conflicts
PYTHONPATH="" python vision_ccs.py

echo "=== Job finished at $(date) with exit code: $? ==="