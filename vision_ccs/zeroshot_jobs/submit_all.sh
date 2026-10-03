#!/bin/bash
# Submit zero-shot jobs. Run with bash (not sbatch), from anywhere:
#   bash zeroshot_jobs/submit_all.sh 00               # smoke test first
#   bash zeroshot_jobs/submit_all.sh                  # then all 12 jobs
#   bash zeroshot_jobs/submit_all.sh 07 10            # only jobs 07 and 10
#   LIMIT=2000 bash zeroshot_jobs/submit_all.sh 07    # capped rerun of job 07
# Every submission is appended to zeroshot_jobs/logs/submitted.txt.
set -e
# job scripts expect vision_ccs/ as the submit dir (logs path, data paths)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p zeroshot_jobs/logs
[ $# -eq 0 ] && set -- 01 02 03 04 05 06 07 08 09 10 11 12

for N in "$@"; do
  F=$(ls zeroshot_jobs/"$N"_*.sh 2>/dev/null | head -1)
  if [ -z "$F" ]; then echo "no job $N in zeroshot_jobs/" >&2; exit 1; fi
  ID=$(sbatch --parsable "$F")
  echo "$(date '+%F %T')  job ${ID%%;*}  $(basename "$F" .sh)${LIMIT:+  LIMIT=$LIMIT}" \
    | tee -a zeroshot_jobs/logs/submitted.txt
done
