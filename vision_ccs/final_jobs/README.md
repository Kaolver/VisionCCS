# Final jobs

The experiment behind the report's main table and figure. For each model and
dataset, on the **same test questions** (grouped 60/40 split, so no image is
in both train and test):

- **zero-shot**: the model's own calibrated Yes/No answer
- **CCS**: unsupervised probe (Burns et al.), at every 4th layer, not only the last
- **logistic regression**: the supervised baseline, with its regularisation tuned
  on a held-out slice of train (the fix the report's baseline never got)
- **shuffled-image control** (job 07): the same with each question paired to
  another image, to test whether the probe needs the image at all

| # | script | model | data | questions | seeds | time limit |
|---|---|---|---|---|---|---|
| 00 | `00_smoke_test.sh` | all | both | 40 | 1 | 00:45:00 |
| 01 | `01_llava_vqa.sh` | LLaVA-1.5-7B | VQAv2 | all | 3 | 02:00:00 |
| 02 | `02_qwen2_vqa.sh` | Qwen2-VL-7B | VQAv2 | all | 3 | 01:45:00 |
| 03 | `03_qwen2_5_vqa.sh` | Qwen2.5-VL-7B | VQAv2 | all | 3 | 01:30:00 |
| 04 | `04_llava_vg.sh` | LLaVA-1.5-7B | Visual Genome | 2000/category | 3 | 01:15:00 |
| 05 | `05_qwen2_vg.sh` | Qwen2-VL-7B | Visual Genome | 2000/category | 3 | 01:15:00 |
| 06 | `06_qwen2_5_vg.sh` | Qwen2.5-VL-7B | Visual Genome | 2000/category | 3 | 01:15:00 |
| 07 | `07_qwen2_5_vqa_shuffled.sh` | Qwen2.5-VL-7B | VQAv2 (shuffled images) | all | 2 | 01:15:00 |

## Run (on Snellius, from `~/VisionCCS/vision_ccs`)

```bash
bash final_jobs/submit_all.sh                        # 00 smoke test + 01-07
squeue -u $USER -o "%.10i %.24j %.8T %.10M %.10l"    # full job names
tail -1 final_jobs/logs/fj00_smoke_test_*.out        # after ~10 min: FINAL SMOKE TEST OK
source _slurm_common.sh && python final_summary.py   # the report tables, any time
```

`final_summary.py` writes `final_report/final_tables.md` (paste into the report)
and `final_results.csv`. The figure `layer_curves.png` needs matplotlib, which
the cluster venv may lack; in that case copy `final_results/` to a laptop and
run the same command there.

If a job runs out of time, submit it again with the same number: finished
zero-shot runs, cached hidden states and finished sweep categories are kept.
