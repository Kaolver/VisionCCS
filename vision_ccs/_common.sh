# Shared preamble for every VisionCCS job. Source it; do not execute it.
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# Locates the code, loads modules, prepares the venv, prints what ran.

# linear_ccs.py reads ./vqav2_mapped.json at import time, so the working
# directory must be the code directory for every entrypoint.
if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/linear_ccs.py" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
elif [ -d "$HOME/VisionCCS/vision_ccs" ]; then
    cd "$HOME/VisionCCS/vision_ccs" || exit 1
fi

if command -v module &> /dev/null; then
    module purge                             2>/dev/null || true
    module load 2023                         2>/dev/null || true
    module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
    module load CUDA/12.1.1                  2>/dev/null || true
fi

# Build the venv once; later jobs reuse it. The marker file means a run that
# died midway through pip does not leave a half-populated venv looking ready.
if [ ! -f venv/.complete ]; then
    echo "=== creating venv ==="
    rm -rf venv
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate pillow tqdm scikit-learn \
        "numpy<2" qwen-vl-utils
    touch venv/.complete
else
    source venv/bin/activate
fi

if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: $(which python) cannot import torch." >&2
    exit 1
fi

echo "host    : $(hostname)"
echo "workdir : $(pwd)"
echo "python  : $(which python)"
python -c "import torch; print('cuda    :', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
command -v nvidia-smi >/dev/null && \
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--------------------------------------------------------------------"
