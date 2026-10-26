#!/bin/bash
# bash text_ccs/jobs/submit_all.sh [job numbers]   (default: 00 to 08)
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p text_ccs/jobs/logs
if [ ! -f text_ccs/data/imdb.jsonl ]; then
  echo "no text_ccs/data/*.jsonl yet: run  pip install pyarrow==17.0.0 && python text_ccs/prepare_data.py  first" >&2
  exit 1
fi
[ $# -eq 0 ] && set -- 00 01 02 03 04 05 06 07 08

for N in "$@"; do
  F=$(ls text_ccs/jobs/"$N"_*.sh 2>/dev/null | head -1)
  if [ -z "$F" ]; then echo "no job $N in text_ccs/jobs/" >&2; exit 1; fi
  # without --export=ALL the environment does not reach jobs on Snellius
  ID=$(sbatch --parsable --export=ALL "$F")
  echo "$(date '+%F %T')  job ${ID%%;*}  $(basename "$F" .sh)" | tee -a text_ccs/jobs/logs/submitted.txt
done
