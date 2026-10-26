#!/bin/bash
#SBATCH --job-name=tc00_smoke_test
#SBATCH --output=text_ccs/jobs/logs/%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Every code path on 20 items x 2 prompts. The last log line says OK or what failed.
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
export PYTHONUNBUFFERED=1
set +e
C=./caches_text_smoke; R=./text_ccs_smoke_results
rm -rf "$C" "$R" ./text_ccs_smoke_report
mkdir -p "$C" "$R"
python text_ccs/test_text_ccs.py 2>&1 | tail -3
E="--limit 20 --max-templates 2 --out-dir $C"
S="--cache-dir $C --seeds 0 --epochs 100 --ntries 2 --positions final --last-layer-positions answer"
for MF in gptj:raw llava:raw llava:chat qwen2:chat qwen2_5:raw qwen2_5_llm:chat vicuna:chat; do
  M=${MF%%:*}; F=${MF##*:}
  echo "=== smoke $M / $F ==="
  python text_ccs/extract_text.py --model $M --format $F $E
  python text_ccs/evaluate_text.py --model $M --format $F $S --out $R/${M}_text-${F}.json
done
python text_ccs/summarize.py --results-dir $R --out-dir ./text_ccs_smoke_report > /dev/null

python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'text_ccs')
import prompts_text as PT
bad = []
for mf in 'gptj:raw llava:raw llava:chat qwen2:chat qwen2_5:raw qwen2_5_llm:chat vicuna:chat'.split():
    m, f = mf.split(':')
    p = Path(f'text_ccs_smoke_results/{m}_text-{f}.json')
    ds = json.loads(p.read_text())['datasets'] if p.exists() else {}
    miss = [d for d in PT.DATASETS if d not in ds or not ds[d]['cells']]
    if miss:
        bad.append(f'{m}/{f}: no result for {miss}')
    man = Path(f'caches_text_smoke/manifest_{m}_{f}.json')
    if man.exists():
        ex = json.loads(man.read_text()).get('imdb', {})
        print(f'{m}/{f} imdb answer token {ex.get("answer_token")!r}, final token {ex.get("final_token")!r}')
if not Path('text_ccs_smoke_report/text_ccs_tables.md').exists():
    bad.append('summarize.py produced no tables')
if bad:
    print('TEXT SMOKE TEST FAILED:\n  ' + '\n  '.join(bad)); sys.exit(1)
print('TEXT SMOKE TEST OK')
PY
