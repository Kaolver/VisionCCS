#!/bin/bash
# Submit the final jobs. Run with bash (not sbatch), from anywhere:
#   bash final_jobs/submit_all.sh                # smoke test 00 + jobs 01-07
#   bash final_jobs/submit_all.sh 03 07          # only jobs 03 and 07 (e.g. a re-run)
# Every submission is appended to final_jobs/logs/submitted.txt.
set -e
# job scripts expect vision_ccs/ as the submit dir (logs path, data paths)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p final_jobs/logs
[ $# -eq 0 ] && set -- 00 01 02 03 04 05 06 07

for N in "$@"; do
  F=$(ls final_jobs/"$N"_*.sh 2>/dev/null | head -1)
  if [ -z "$F" ]; then echo "no job $N in final_jobs/" >&2; exit 1; fi
  # --export=ALL explicitly: on Snellius the environment does not reach jobs without it
  ID=$(sbatch --parsable --export=ALL "$F")
  echo "$(date '+%F %T')  job ${ID%%;*}  $(basename "$F" .sh)" | tee -a final_jobs/logs/submitted.txt
done
