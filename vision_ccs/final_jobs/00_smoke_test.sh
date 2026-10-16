#!/bin/bash
#SBATCH --job-name=fj00_smoke_test
#SBATCH --output=final_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 00 -- exercises every step of jobs 01-07 on 40 questions, ~10 min.
# Submit it together with 01-07: it finishes long before they reach their
# layer sweep, so a code problem it finds can be fixed (git pull) in time.
# The last line of the log says FINAL SMOKE TEST OK or lists what failed.
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
set +e   # keep going: the verdict at the end lists every failure
C=./caches_smoke; R=./smoke_results
rm -rf "$C" "$R" ./smoke_report
mkdir -p "$C/vqa" "$C/vg" "$R"
python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json
E="--limit 40 --layer-stride 8"
S="--positions final --seeds 42 --grouped --selection val_consistency --epochs 50 --ntries 2 --pick-layer"
for M in llava qwen2 qwen2_5; do
  python extract.py --model $M --categories object_detection $E --out-dir $C/vqa
  python extract.py --model $M --vqa-json ./vg_mapped.json --categories object $E --out-dir $C/vg
  python layer_sweep.py --cache-dir $C/vqa --model $M --categories object_detection $S \
      --zeroshot-dir ./zeroshot_report --zeroshot-tags _vqa_noinstr _vqa --out $R/sweep_${M}_vqa.json
  python layer_sweep.py --cache-dir $C/vg --model $M --categories object $S \
      --zeroshot-dir ./zeroshot_report --zeroshot-tags _vg_noinstr _vg --out $R/sweep_${M}_vg.json
done
python extract.py --model qwen2_5 --categories object_detection $E --shuffle-images --out-dir $C/vqa
python layer_sweep.py --cache-dir $C/vqa --model qwen2_5 --categories object_detection --shuffled $S \
    --out $R/sweep_qwen2_5_vqa_shuffled.json
python final_summary.py --results-dir $R --out-dir ./smoke_report > /dev/null

python - <<'EOF'
import json, sys
from pathlib import Path
bad = []
for m in ('llava', 'qwen2', 'qwen2_5'):
    for d in ('vqa', 'vg'):
        f = Path(f'smoke_results/sweep_{m}_{d}.json')
        cells = json.loads(f.read_text())['cells'] if f.exists() else {}
        if not any(c['grid'] for c in cells.values()):
            bad.append(f'{m} / {d}: no sweep result (see extract.py / layer_sweep.py output above)')
f = Path('smoke_results/sweep_qwen2_5_vqa_shuffled.json')
if not (f.exists() and any(c['grid'] for c in json.loads(f.read_text())['cells'].values())):
    bad.append('qwen2_5 / vqa shuffled: no sweep result')
if not Path('smoke_report/final_tables.md').exists():
    bad.append('final_summary.py produced no tables')
if bad:
    print('FINAL SMOKE TEST FAILED:\n  ' + '\n  '.join(bad))
    sys.exit(1)
print('FINAL SMOKE TEST OK: extraction, layer sweep, zero-shot join and summary all ran')
EOF
