#!/bin/bash
#SBATCH --job-name=ccs-smoke-test
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=32G

# Fast smoke test for Phase 1 extraction on a GPU node

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/extract.py" ]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
elif [ -d "$HOME/VisionCCS/vision_ccs" ]; then
  cd "$HOME/VisionCCS/vision_ccs" || exit 1
elif [ -d "$HOME/vision_ccs" ]; then
  cd "$HOME/vision_ccs" || exit 1
fi

if command -v module &> /dev/null; then
  module purge 2>/dev/null || true
  module load 2023 2>/dev/null || true
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
  module load CUDA/12.1.1 2>/dev/null || true
fi

if [ -f "venv_ccs/bin/activate" ]; then
  source venv_ccs/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "../venv_ccs/bin/activate" ]; then
  source ../venv_ccs/bin/activate
elif [ -f "../venv/bin/activate" ]; then
  source ../venv/bin/activate
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python extract.py --model qwen2 --limit 5 --categories object_detection --out-dir /tmp/smoke
