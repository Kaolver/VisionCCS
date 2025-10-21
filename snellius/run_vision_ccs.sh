#!/bin/bash
#SBATCH --job-name=snellius-vision-ccs
#SBATCH --output=%x_%j.out    
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681
#SBATCH --mem=32G

# Load required modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Create and activate virtual environment (if not exists)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with system site packages..."
    python -m venv --system-site-packages venv
    source venv/bin/activate
    echo "Installing accelerate for model loading..."
    pip install --quiet accelerate
else
    source venv/bin/activate
fi

echo "=== Checking if vision_ccs.py exists ==="
ls -lh vision_ccs.py

# Run the VisionCCS script
echo "=== Running vision_ccs.py ==="
python vision_ccs.py

# Exit status
echo "=== Job finished at $(date) with exit code: $? ==="
