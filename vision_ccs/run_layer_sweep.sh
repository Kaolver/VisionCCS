#!/bin/bash
#SBATCH --job-name=ccs-layer-sweep
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=96G

# Phase 2.4: CCS accuracy vs LAYER and token POSITION, on the Phase 1 caches.
#
# Everything before this probed the final layer -- the readout layer, which
# zero-shot IS through the LM head. That is the likeliest reason CCS agreed with
# zero-shot on 90.9% of items. Truth-related structure is expected mid-stack.
#
# REPORTING RULE: the full curve is the result. --pick-layer names a single
# layer using the label-free consistency criterion and prints what that honest
# choice cost against the oracle layer. Never quote the max over layers.

source "$(dirname "$0")/_slurm_common.sh"

if [ ! -d ./caches_v3 ]; then
  echo "ERROR: ./caches_v3 not found -- run run_extract_v3.sh first"; exit 1
fi
ls -lh ./caches_v3

for MODEL in qwen2 llava; do
  if ! ls ./caches_v3/hs_${MODEL}_*.npz >/dev/null 2>&1; then
    echo "no caches for $MODEL, skipping"; continue
  fi

  echo ""
  echo "######################################################################"
  echo "# $MODEL : real images"
  echo "######################################################################"
  python layer_sweep.py --cache-dir ./caches_v3 --model "$MODEL" \
      --seeds 42 1 2 --selection val_consistency --skip-logreg --pick-layer \
      --out "./layer_sweep_${MODEL}.json"

  echo ""
  echo "######################################################################"
  echo "# $MODEL : SHUFFLED-IMAGE CONTROL (B2)"
  echo "# If the curve here tracks the one above, the probe never needed the"
  echo "# image and we are reading a VQAv2 language prior."
  echo "######################################################################"
  python layer_sweep.py --cache-dir ./caches_v3 --model "$MODEL" --shuffled \
      --seeds 42 1 2 --selection val_consistency --skip-logreg \
      --out "./layer_sweep_${MODEL}_shuffled.json"
done

echo ""
echo "=== outputs ==="
ls -lh ./layer_sweep_*.json
