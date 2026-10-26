# Text-only CCS

Runs the original CCS protocol (Burns et al. 2022) on the paper's text datasets
with our vision models, to tell whether our results differ from the paper
because of the models or because of the vision setup.

## Reading the results

| step | compares | if they differ |
|---|---|---|
| 1 | GPT-J here vs GPT-J in the paper | the pipeline is off; fix it before reading the rest |
| 2 | our VLMs on text vs GPT-J and the paper | the models are a cause |
| 3 | each VLM vs its language model | vision training changed what CCS finds |
| 4 | VLM raw text, chat text, images | the vision setup is a cause |

Paper reference (Table 1), GPT-J: calibrated 0-shot 56.0, CCS 61.7, LR 78.1.
Mean over the paper's 5 models: 67.2, 71.2, 83.7. Expected pattern:
0-shot <= CCS <= LR.

Job 08 scores the `final_jobs` vision caches with the same evaluator, so a text
vs image gap cannot come from two different CCS implementations.

## Setup

Models: `gptj` (GPT-J-6B), `llava`, `qwen2`, `qwen2_5` (no image), `vicuna`
(LLaVA-1.5's language model), `qwen2_llm`, `qwen2_5_llm` (Qwen2 and
Qwen2.5-7B-Instruct).

Formats: `raw` is question + `\n` + answer + EOS, as in `ccs/utils.py`. `chat`
puts the same text in the user turn of the chat template, as the vision
pipeline does.

Datasets: IMDB, Amazon, AG News, DBpedia-14, RTE, QNLI, COPA, BoolQ, PIQA.
1000 balanced items each (COPA 500), split 60/40. Story Cloze is left out.

Protocol: per-prompt normalization, CCS with 10 restarts of 1000 epochs at
lr 0.01, LR on the concatenated pair with C = 1 and with C tuned on train,
calibrated zero-shot from answer log-probs. Scored at every 4th layer at the
last token, and at the last answer token on the last layer.

Prompts: 5 to 8 per dataset (the paper used 8 to 13). The ones printed in the
paper's Appendix I are verbatim (`burns`); the rest paraphrase promptsource
(`ps-style`). Check these first if step 1 is off.

## Run (Snellius, from `~/VisionCCS/vision_ccs`)

```bash
pip install pyarrow
python text_ccs/prepare_data.py
python text_ccs/test_text_ccs.py
TEXT_CACHE_DIR=/scratch-shared/$USER/caches_text bash text_ccs/jobs/submit_all.sh
python text_ccs/summarize.py
```

| job | runs | time limit |
|---|---|---|
| 00 | smoke test, 20 items x 2 prompts | 01:30 |
| 01 | GPT-J, raw | 02:30 |
| 02 to 04 | LLaVA, Qwen2-VL, Qwen2.5-VL, raw and chat | 04:00 |
| 05 to 07 | Vicuna, Qwen2-7B, Qwen2.5-7B-Instruct, raw | 02:30 |
| 08 | evaluator on `caches_final/` | 03:00 |

Time limits are estimates. Resubmitting a job keeps finished datasets. Hidden
states take about 15 GB per model and format; model downloads about 70 GB.

GPT-J and Vicuna only ship `.bin` weights, which transformers 4.57 will not load
under torch 2.5.1, so they load from the hub's safetensors conversions
(`refs/pr/40`, `refs/pr/13`).

Results: `text_ccs_results/*.json` and `text_ccs_report/text_ccs_tables.md`.
