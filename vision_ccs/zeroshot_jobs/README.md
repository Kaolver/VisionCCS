# Zero-shot jobs

One Slurm script per job, so each job's exact command is readable on its own.
Every job writes into `vision_ccs/zeroshot_report/`; its log goes to
`zeroshot_jobs/logs/zs<NN>_<name>_<jobid>.out`.

| # | script | model | dataset | prompt | questions | time limit |
|---|---|---|---|---|---|---|
| 00 | `00_smoke_test.sh` | all 3 | both | both | 5 per category | 01:00:00 |
| 01 | `01_llava_vqa_noinstr.sh` | LLaVA-1.5-7B | VQAv2 | matched to CCS | 5,648 questions | 01:30:00 |
| 02 | `02_llava_vqa_instr.sh` | LLaVA-1.5-7B | VQAv2 | with "Answer yes or no." instruction | 5,648 questions | 01:30:00 |
| 03 | `03_qwen2_vqa_noinstr.sh` | Qwen2-VL-7B | VQAv2 | matched to CCS | 5,648 questions | 01:30:00 |
| 04 | `04_qwen2_vqa_instr.sh` | Qwen2-VL-7B | VQAv2 | with "Answer yes or no." instruction | 5,648 questions | 01:30:00 |
| 05 | `05_qwen2_5_vqa_noinstr.sh` | Qwen2.5-VL-7B | VQAv2 | matched to CCS | 5,648 questions | 01:30:00 |
| 06 | `06_qwen2_5_vqa_instr.sh` | Qwen2.5-VL-7B | VQAv2 | with "Answer yes or no." instruction | 5,648 questions | 01:30:00 |
| 07 | `07_llava_vg_noinstr.sh` | LLaVA-1.5-7B | Visual Genome | matched to CCS | 36,000 questions | 06:00:00 |
| 08 | `08_llava_vg_instr.sh` | LLaVA-1.5-7B | Visual Genome | with "Answer yes or no." instruction | 36,000 questions | 06:00:00 |
| 09 | `09_qwen2_vg_noinstr.sh` | Qwen2-VL-7B | Visual Genome | matched to CCS | 36,000 questions | 06:00:00 |
| 10 | `10_qwen2_vg_instr.sh` | Qwen2-VL-7B | Visual Genome | with "Answer yes or no." instruction | 36,000 questions | 06:00:00 |
| 11 | `11_qwen2_5_vg_noinstr.sh` | Qwen2.5-VL-7B | Visual Genome | matched to CCS | 36,000 questions | 06:00:00 |
| 12 | `12_qwen2_5_vg_instr.sh` | Qwen2.5-VL-7B | Visual Genome | with "Answer yes or no." instruction | 36,000 questions | 06:00:00 |

The prompt "matched to CCS" leaves out the "Answer yes or no." hint, so the
model sees the same question CCS is trained on; it is the fair comparison. The
other prompt is the usual zero-shot setup.

## Run (on Snellius, from `~/VisionCCS/vision_ccs`)

```bash
bash zeroshot_jobs/submit_all.sh 00                 # smoke test; wait for it
tail -1 zeroshot_jobs/logs/zs00_smoke_test_*.out    # must say SMOKE TEST OK
bash zeroshot_jobs/submit_all.sh                    # all 12 in parallel
squeue -u $USER -o "%.10i %.28j %.8T %.10M %.10l"   # full job names
python collect_zeroshot.py --out-dir ./zeroshot_report   # the table, any time
```

`collect_zeroshot.py` needs the venv: on a login node, `source _slurm_common.sh` first.

## If a Visual Genome job is too slow

About 30 min after it starts, `tail -3` its log. If it has not reached
`object: 2500/12000`, it will not finish in 6 h: `scancel` it and resubmit it
capped at 2,000 questions per category, e.g.
`LIMIT=2000 bash zeroshot_jobs/submit_all.sh 07`. The capped run overwrites
that job's files; the table's `n` column shows the cap.
