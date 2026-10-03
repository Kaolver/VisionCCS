#!/bin/bash
#SBATCH --job-name=zs00_smoke_test
#SBATCH --output=zeroshot_jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Job 00 -- run this first, alone, and wait for it.
# Every model x dataset x prompt at 5 questions per category. Also downloads each
# model once, so the 12 parallel jobs don't all download at the same time.
# The last line of the log says SMOKE TEST OK or lists what failed.
#
# Submit from vision_ccs/:  bash zeroshot_jobs/submit_all.sh 00
# modules + venv, so the check below has a python; it also cd's to vision_ccs/
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
LIMIT=5 OUT=./zeroshot_smoke bash ./run_zeroshot_all.sh

python - <<'EOF'
import json, sys
from pathlib import Path
out, bad = Path('zeroshot_smoke'), []
grid = {'vqa': ['object_detection', 'attribute_recognition', 'spatial_recognition'],
        'vg': ['object', 'attribute', 'spatial']}
for m in ('llava', 'qwen2', 'qwen2_5'):
    for d, cats in grid.items():
        for v in ('_noinstr', ''):
            f = out / f'zeroshot_{m}_{d}{v}_summary.json'
            if not f.exists():
                bad.append(f'{m} / {d}{v}: did not finish (see errors above)')
                continue
            s = json.loads(f.read_text())
            for c in cats:
                r = s.get(c)
                if r is None or r['n'] != 5 or r['skipped']:
                    bad.append(f'{m} / {d}{v} / {c}: ' + ('no result' if r is None
                               else f"n={r['n']} skipped={r['skipped']}"))
if bad:
    print('SMOKE TEST FAILED -- do not submit the 12 jobs yet:\n  ' + '\n  '.join(bad))
    sys.exit(1)
print('SMOKE TEST OK: all 12 combinations ran on 5 questions each, nothing skipped')
EOF
