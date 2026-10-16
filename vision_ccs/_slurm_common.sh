# Shared preamble for every VisionCCS batch job. Source it, do not execute it.
#   source "$(dirname "${BASH_SOURCE[0]}")/_slurm_common.sh"
# Locates the project, loads modules, activates the venv, and prints what ran.

if [ -n "$SLURM_SUBMIT_DIR" ] && [ -f "$SLURM_SUBMIT_DIR/reanalysis.py" ]; then
  cd "$SLURM_SUBMIT_DIR" || exit 1
elif [ -d "$HOME/VisionCCS/vision_ccs" ]; then
  cd "$HOME/VisionCCS/vision_ccs" || exit 1
elif [ -d "$HOME/vision_ccs" ]; then
  cd "$HOME/vision_ccs" || exit 1
fi

if command -v module &> /dev/null; then
  module purge            2>/dev/null || true
  module load 2023        2>/dev/null || true
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
  module load CUDA/12.1.1 2>/dev/null || true
fi

for V in venv_ccs venv ../venv_ccs ../venv; do
  if [ -f "$V/bin/activate" ]; then source "$V/bin/activate"; break; fi
done

if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: Active python ($(which python)) cannot import torch. Check venv activation." >&2
  exit 1
fi

echo "host      : $(hostname)"
echo "workdir   : $(pwd)"
echo "python    : $(which python)"
echo "git       : $(git rev-parse --short HEAD 2>/dev/null || echo n/a) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "----------------------------------------------------------------------"
