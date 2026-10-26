# Sourced by the text-CCS jobs after setting MODEL and FORMATS.
for DIR in "${SLURM_SUBMIT_DIR}" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done
set -e
export PYTHONUNBUFFERED=1
C=${TEXT_CACHE_DIR:-./caches_text}
R=./text_ccs_results
mkdir -p "$C" "$R"
if [ ! -f text_ccs/data/imdb.jsonl ]; then
  echo "ERROR: no text_ccs/data/*.jsonl, run 'python text_ccs/prepare_data.py' first" >&2
  exit 1
fi
for FMT in $FORMATS; do
  echo "=== $MODEL / $FMT: hidden states ==="
  python text_ccs/extract_text.py --model "$MODEL" --format "$FMT" --out-dir "$C"
  echo "=== $MODEL / $FMT: CCS ==="
  python text_ccs/evaluate_text.py --model "$MODEL" --format "$FMT" --cache-dir "$C" \
      --positions final --last-layer-positions answer --seeds 0 1 2 --resume \
      --out "$R/${MODEL}_text-${FMT}.json"
done
python text_ccs/summarize.py --results-dir "$R" --out-dir ./text_ccs_report
echo "=== done $MODEL ($FORMATS) ==="
