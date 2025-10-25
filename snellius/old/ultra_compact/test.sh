#!/bin/bash
#SBATCH --job-name=snellius-ultra-compact
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681

# Load CUDA module (adjust version to match your system)
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

# Create virtual environment if it doesn't exist (inherit system packages)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with system site packages..."
    python -m venv --system-site-packages venv
fi

# Activate virtual environment
source venv/bin/activate

echo "=== Starting job at $(date) ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Checking if ultra_compact.py exists:"
ls -lh ultra_compact.py
echo "=== Running ultra_compact.py ==="
 
python ultra_compact.py

echo "=== Job finished at $(date) with exit code: $? ==="