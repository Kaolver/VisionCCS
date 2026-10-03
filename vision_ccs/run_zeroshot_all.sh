#!/bin/bash
#SBATCH --job-name=zs
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_a100
#SBATCH --mem=64G

# Zero-shot Yes/No baseline over a grid of model x dataset x prompt variant.
# Every axis is an env var, so one script serves the smoke test and each
# parallel job:
#
#   MODELS    llava qwen2 qwen2_5   (default: all three)
#   DATASETS  vqa vg                (default: both)
#   VARIANTS  noinstr instr         (default: both, noinstr first)
#               noinstr = prompt-matched to CCS (no "Answer yes or no.")
#               instr   = with the "Answer yes or no." hint
#   LIMIT     items per category    (default: all; 5 for a smoke test)
#   OUT       output directory      (default: ./zeroshot_report)
#
# For the report, don't submit this directly: zeroshot_jobs/ has one script per
# job (00 = smoke test, which calls this with LIMIT=5; 01-12 = the parallel
# runs) and submit_all.sh. See zeroshot_jobs/README.md.
#
# Outputs: $OUT/zeroshot_{model}_{dataset}[_noinstr]_{category}.npz + _summary.json
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

set -e
export PYTHONUNBUFFERED=1   # progress lines reach the Slurm log as they happen
MODELS="${MODELS:-llava qwen2 qwen2_5}"
DATASETS="${DATASETS:-vqa vg}"
VARIANTS="${VARIANTS:-noinstr instr}"
OUT="${OUT:-./zeroshot_report}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
mkdir -p "$OUT"
# one failing combination must not hide the others (the smoke test relies on
# seeing every failure in one run), so record failures and exit non-zero at the end
FAILED=""

echo "models=$MODELS  datasets=$DATASETS  variants=$VARIANTS  limit=${LIMIT:-all}  out=$OUT"

for DATASET in $DATASETS; do
  case "$DATASET" in
    vqa) DATA_ARGS="--vqa-json ./vqav2_mapped.json --categories object_detection attribute_recognition spatial_recognition" ;;
    vg)  python vg_to_vqa.py --vg-dir ../vg --out ./vg_mapped.json
         DATA_ARGS="--vqa-json ./vg_mapped.json --categories object attribute spatial" ;;
    *)   echo "unknown dataset $DATASET (expected vqa or vg)" >&2; exit 1 ;;
  esac

  for MODEL in $MODELS; do
    for VARIANT in $VARIANTS; do
      case "$VARIANT" in
        noinstr) VAR_ARGS="--no-instruction"; TAG="_${DATASET}_noinstr" ;;
        instr)   VAR_ARGS="";                 TAG="_${DATASET}" ;;
        *)       echo "unknown variant $VARIANT (expected noinstr or instr)" >&2; exit 1 ;;
      esac
      echo ""
      echo "=== $MODEL / $DATASET / $VARIANT  ($(date '+%H:%M:%S')) ==="
      python zero_shot.py --model "$MODEL" $DATA_ARGS $VAR_ARGS $LIMIT_ARG \
          --tag "$TAG" --out-dir "$OUT" || FAILED="$FAILED $MODEL/$DATASET/$VARIANT"
    done
  done
done

echo ""
echo "=== done $(date '+%H:%M:%S'); outputs ==="
ls -lh "$OUT"
if [ -n "$FAILED" ]; then
  echo "FAILED:$FAILED" >&2
  exit 1
fi
