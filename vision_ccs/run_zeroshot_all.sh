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
# Smoke test (also downloads all three models once, so parallel jobs don't race):
#   sbatch --time=01:00:00 --export=ALL,LIMIT=5,OUT=./zeroshot_smoke run_zeroshot_all.sh
# One parallel job:
#   sbatch --job-name=zs-qwen2-vg-noinstr \
#       --export=ALL,MODELS=qwen2,DATASETS=vg,VARIANTS=noinstr run_zeroshot_all.sh
# Then: python collect_zeroshot.py --out-dir ./zeroshot_report
#
# Outputs: $OUT/zeroshot_{model}_{dataset}[_noinstr]_{category}.npz + _summary.json
for DIR in "${SLURM_SUBMIT_DIR}" "." "$(dirname "$0")" "$HOME/VisionCCS/vision_ccs"; do
  if [ -n "$DIR" ] && [ -f "$DIR/_slurm_common.sh" ]; then
    source "$DIR/_slurm_common.sh"
    break
  fi
done

set -e
MODELS="${MODELS:-llava qwen2 qwen2_5}"
DATASETS="${DATASETS:-vqa vg}"
VARIANTS="${VARIANTS:-noinstr instr}"
OUT="${OUT:-./zeroshot_report}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
mkdir -p "$OUT"

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
          --tag "$TAG" --out-dir "$OUT"
    done
  done
done

echo ""
echo "=== done $(date '+%H:%M:%S'); outputs ==="
ls -lh "$OUT"
