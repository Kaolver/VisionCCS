#!/bin/bash
#SBATCH --job-name=ccs-smoke-test
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=32G

# Pre-flight: exercises every path the expensive runs depend on at --limit 5.
# Run before any long job.
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

set -e
OUT=./smoke_out
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== 0. unit tests (no GPU needed; must pass before anything else) ==="
python test_reanalysis.py

echo ""
echo "=== 1. extraction: single template ==="
python extract.py --model qwen2 --limit 5 --categories object_detection \
    --out-dir "$OUT/caches"

echo ""
echo "=== 2. extraction: multi-template (rows = items x templates) ==="
python extract.py --model qwen2 --limit 5 --categories object_detection \
    --templates all --out-dir "$OUT/caches"

echo ""
echo "=== 3. extraction: shuffled-image and banner-distractor controls ==="
python extract.py --model qwen2 --limit 5 --categories object_detection \
    --shuffle-images --out-dir "$OUT/caches"
python extract.py --model qwen2 --limit 5 --categories object_detection \
    --distractor banner --out-dir "$OUT/caches"

echo ""
echo "=== 4. layer sweep on each cache variant ==="
python layer_sweep.py --cache-dir "$OUT/caches" --model qwen2 \
    --categories object_detection --seeds 42 --epochs 50 --ntries 2 \
    --skip-logreg --pick-layer --out "$OUT/ls.json"
python layer_sweep.py --cache-dir "$OUT/caches" --model qwen2 --distractor banner \
    --categories object_detection --seeds 42 --epochs 50 --ntries 2 \
    --skip-logreg --out "$OUT/ls_banner.json"
python layer_sweep.py --cache-dir "$OUT/caches" --model qwen2 --templates all \
    --categories object_detection --seeds 42 --epochs 50 --ntries 2 \
    --skip-logreg --out "$OUT/ls_templates.json"

echo ""
echo "=== 5. zero-shot, both prompt variants ==="
python zero_shot.py --model qwen2 --limit 5 --categories object_detection \
    --out-dir "$OUT/zs"
python zero_shot.py --model qwen2 --limit 5 --categories object_detection \
    --no-instruction --tag _noinstr --out-dir "$OUT/zs"

echo ""
echo "=== 6. transfer (needs >=2 categories; expected to no-op at --limit) ==="
python transfer.py --cache-dir "$OUT/caches" --cache-version v3 --model qwen2 \
    --seeds 42 --epochs 50 --ntries 2 --out "$OUT/tr.json" || \
    echo "(transfer needs >=2 categories extracted -- fine for a smoke test)"

echo ""
echo "=== smoke test complete: all paths exercised ==="
ls -lhR "$OUT"
