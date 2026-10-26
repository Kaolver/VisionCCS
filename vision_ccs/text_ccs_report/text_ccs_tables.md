# Original CCS on the vision-CCS models, text only

Accuracy %, mean over datasets, prompts and seeds. CCS s.d. across prompts in parentheses. Paper columns: Burns et al. 2022, Table 1. Results: gptj/text-raw, llava/text-chat, llava/text-raw, qwen2/text-chat, qwen2/text-raw, qwen2_5/text-chat, qwen2_5/text-raw, qwen2_llm/text-raw.

## Reading

- **Reproduction (GPT-J, raw, last layer, last answer token):** CCS 61.8 vs 61.7 in the paper (+0.1), LR 74.9 vs 78.1, calibrated 0-shot 57.7 vs 56.0. Within 5 points of the paper.
- **Reproduction (GPT-J, raw, last layer, EOS token):** CCS 52.4 vs 61.7 in the paper (-9.3), LR 65.6 vs 78.1, calibrated 0-shot 57.7 vs 56.0. More than 5 points off the paper.
- **LLaVA-1.5-7B, text only:** CCS 72.7, calibrated 0-shot 72.7, LR 82.8 (tuned 83.8). 0-shot <= CCS <= LR does not hold. 
- **Qwen2-VL-7B, text only:** CCS 88.4, calibrated 0-shot 86.4, LR 87.7 (tuned 88.5). 0-shot <= CCS <= LR holds. Its language model Qwen2-7B-Instruct: CCS 84.5. 
- **Qwen2.5-VL-7B, text only:** CCS 89.4, calibrated 0-shot 86.3, LR 88.8 (tuned 90.0). 0-shot <= CCS <= LR holds. 

## Paper protocol: raw text, last layer, last answer token (Algorithm 1)

| Method | GPT-J-6B | LLaVA-1.5-7B | Qwen2-VL-7B | Qwen2.5-VL-7B | Qwen2-7B-Instruct | GPT-J | paper mean (5 models) |
|---|---|---|---|---|---|---|---|
| 0-shot | 55.0 | 62.7 | 81.0 | 79.2 | 83.9 | 53.2 | 62.8 |
| Calibrated 0-shot | 57.7 | 72.7 | 86.4 | 86.3 | 87.3 | 56.0 | 67.2 |
| CCS | 61.8 (2.3) | 72.7 (4.6) | 88.4 (3.8) | 89.4 (3.3) | 84.5 (8.4) | 61.7 | 71.2 |
| LR (ceiling, C=1) | 74.9 | 82.8 | 87.7 | 88.8 | 89.3 | 78.1 | 83.7 |
| LR (C tuned on train) | 75.1 | 83.8 | 88.5 | 90.0 | 90.2 | - | - |

Datasets per model: GPT-J-6B: 9 datasets; LLaVA-1.5-7B: 9 datasets; Qwen2-VL-7B: 9 datasets; Qwen2.5-VL-7B: 9 datasets; Qwen2-7B-Instruct: 9 datasets.

## Raw text, last layer, appended EOS token (released code)

| Method | GPT-J-6B | LLaVA-1.5-7B | Qwen2-VL-7B | Qwen2.5-VL-7B | Qwen2-7B-Instruct | GPT-J | paper mean (5 models) |
|---|---|---|---|---|---|---|---|
| 0-shot | 55.0 | 62.7 | 81.0 | 79.2 | 83.9 | 53.2 | 62.8 |
| Calibrated 0-shot | 57.7 | 72.7 | 86.4 | 86.3 | 87.3 | 56.0 | 67.2 |
| CCS | 52.4 (1.3) | 67.3 (3.6) | 86.9 (3.6) | 81.6 (9.9) | 86.7 (5.3) | 61.7 | 71.2 |
| LR (ceiling, C=1) | 65.6 | 76.0 | 88.6 | 89.4 | 89.3 | 78.1 | 83.7 |
| LR (C tuned on train) | 65.7 | 76.8 | 89.8 | 90.5 | 90.4 | - | - |

Datasets per model: GPT-J-6B: 9 datasets; LLaVA-1.5-7B: 9 datasets; Qwen2-VL-7B: 9 datasets; Qwen2.5-VL-7B: 9 datasets; Qwen2-7B-Instruct: 9 datasets.

## Chat-template format, last layer, last answer token

| Method | LLaVA-1.5-7B | Qwen2-VL-7B | Qwen2.5-VL-7B | GPT-J | paper mean (5 models) |
|---|---|---|---|---|---|
| 0-shot | 61.9 | 78.3 | 70.4 | 53.2 | 62.8 |
| Calibrated 0-shot | 71.5 | 85.2 | 82.8 | 56.0 | 67.2 |
| CCS | 75.1 (4.0) | 88.9 (3.1) | 87.4 (4.0) | 61.7 | 71.2 |
| LR (ceiling, C=1) | 83.6 | 88.4 | 89.5 | 78.1 | 83.7 |
| LR (C tuned on train) | 84.6 | 89.4 | 90.7 | - | - |

Datasets per model: LLaVA-1.5-7B: 9 datasets; Qwen2-VL-7B: 9 datasets; Qwen2.5-VL-7B: 9 datasets.

## Chat-template format, last layer, last token (as the vision pipeline)

| Method | LLaVA-1.5-7B | Qwen2-VL-7B | Qwen2.5-VL-7B | GPT-J | paper mean (5 models) |
|---|---|---|---|---|---|
| 0-shot | 61.9 | 78.3 | 70.4 | 53.2 | 62.8 |
| Calibrated 0-shot | 71.5 | 85.2 | 82.8 | 56.0 | 67.2 |
| CCS | 68.5 (2.9) | 87.0 (5.8) | 89.8 (1.9) | 61.7 | 71.2 |
| LR (ceiling, C=1) | 76.7 | 89.9 | 90.1 | 78.1 | 83.7 |
| LR (C tuned on train) | 77.6 | 90.8 | 91.2 | - | - |

Datasets per model: LLaVA-1.5-7B: 9 datasets; Qwen2-VL-7B: 9 datasets; Qwen2.5-VL-7B: 9 datasets.

## Per dataset: CCS, text-raw, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-J-6B | 51.0 | 55.6 | 52.2 | 52.1 | 50.8 | 54.1 | 51.8 | 51.7 | 52.1 | 52.4 |
| LLaVA-1.5-7B | 52.6 | 92.0 | 67.3 | 62.7 | 75.1 | 91.5 | 62.4 | 51.3 | 50.8 | 67.3 |
| Qwen2-VL-7B | 93.4 | 95.8 | 74.0 | 92.5 | 98.9 | 95.8 | 71.3 | 84.6 | 75.9 | 86.9 |
| Qwen2.5-VL-7B | 93.7 | 60.0 | 80.0 | 92.5 | 83.0 | 95.6 | 62.7 | 85.4 | 81.3 | 81.6 |
| Qwen2-7B-Instruct | 93.1 | 95.7 | 72.0 | 90.1 | 98.6 | 88.1 | 74.8 | 86.1 | 82.2 | 86.7 |

## Per dataset: LR (tuned C), text-raw, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-J-6B | 56.9 | 94.7 | 52.8 | 51.3 | 60.3 | 93.1 | 52.7 | 75.0 | 54.4 | 65.7 |
| LLaVA-1.5-7B | 86.1 | 92.3 | 66.7 | 64.5 | 88.2 | 93.7 | 59.0 | 74.0 | 67.0 | 76.8 |
| Qwen2-VL-7B | 94.3 | 96.6 | 79.3 | 91.2 | 99.4 | 95.7 | 76.8 | 89.0 | 85.6 | 89.8 |
| Qwen2.5-VL-7B | 95.0 | 96.2 | 81.0 | 92.2 | 99.3 | 95.5 | 78.7 | 90.6 | 86.2 | 90.5 |
| Qwen2-7B-Instruct | 93.9 | 96.7 | 80.0 | 93.8 | 98.7 | 95.8 | 80.3 | 89.4 | 84.8 | 90.4 |

## Per dataset: Calibrated 0-shot, text-raw, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-J-6B | 52.5 | 67.4 | 53.4 | 58.2 | 48.1 | 76.0 | 61.5 | 50.3 | 51.7 | 57.7 |
| LLaVA-1.5-7B | 82.5 | 79.1 | 73.7 | 67.3 | 76.1 | 82.6 | 69.5 | 58.4 | 65.2 | 72.7 |
| Qwen2-VL-7B | 92.7 | 91.3 | 79.1 | 77.8 | 96.1 | 94.2 | 78.5 | 83.8 | 84.2 | 86.4 |
| Qwen2.5-VL-7B | 91.0 | 88.4 | 80.6 | 79.5 | 95.2 | 92.2 | 79.4 | 84.9 | 85.7 | 86.3 |
| Qwen2-7B-Instruct | 93.4 | 89.3 | 80.2 | 84.8 | 97.2 | 93.9 | 80.0 | 83.4 | 83.5 | 87.3 |

## Per dataset: CCS, text-chat, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 51.7 | 92.4 | 61.2 | 69.7 | 80.9 | 93.4 | 63.0 | 52.4 | 51.5 | 68.5 |
| Qwen2-VL-7B | 85.9 | 95.0 | 79.0 | 78.9 | 98.9 | 95.0 | 77.5 | 87.9 | 84.9 | 87.0 |
| Qwen2.5-VL-7B | 93.8 | 95.9 | 82.4 | 92.6 | 99.1 | 95.6 | 77.7 | 86.0 | 85.4 | 89.8 |

## Per dataset: LR (tuned C), text-chat, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 88.2 | 93.2 | 66.0 | 65.1 | 90.0 | 93.6 | 58.7 | 74.8 | 68.8 | 77.6 |
| Qwen2-VL-7B | 94.3 | 96.4 | 81.0 | 93.2 | 99.2 | 95.0 | 79.8 | 90.7 | 87.3 | 90.8 |
| Qwen2.5-VL-7B | 94.7 | 96.5 | 82.7 | 92.3 | 99.4 | 95.7 | 80.8 | 90.7 | 88.0 | 91.2 |

## Per dataset: Calibrated 0-shot, text-chat, last layer

| Model | ag_news | amazon_polarity | boolq | copa | dbpedia_14 | imdb | piqa | qnli | rte | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 80.9 | 78.4 | 73.9 | 65.7 | 66.8 | 82.6 | 69.5 | 59.9 | 65.4 | 71.5 |
| Qwen2-VL-7B | 91.5 | 89.0 | 80.3 | 71.9 | 94.2 | 93.7 | 77.3 | 84.2 | 85.0 | 85.2 |
| Qwen2.5-VL-7B | 85.2 | 87.1 | 79.5 | 69.6 | 89.0 | 92.1 | 74.3 | 83.8 | 85.0 | 82.8 |

## Same evaluator, text vs images (last layer, last answer token)

| Model | text raw CCS | text raw LR | text chat CCS | text chat LR | VQAv2 CCS | VQAv2 LR | VG CCS | VG LR |
|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 72.7 | 83.8 | 75.1 | 84.6 | - | - | - | - |
| Qwen2-VL-7B | 88.4 | 88.5 | 88.9 | 89.4 | - | - | - | - |
| Qwen2.5-VL-7B | 89.4 | 90.0 | 87.4 | 90.7 | - | - | - | - |

## Layer curves (mean over datasets)

**GPT-J-6B, text-raw** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 51.4 | 51.9 | 52.3 | 52.2 | 52.2 | 52.0 | 52.4 |
| LR (tuned C) | 61.1 | 65.6 | 68.2 | 67.8 | 66.8 | 66.0 | 65.7 |

**LLaVA-1.5-7B, text-chat** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|
| CCS | 51.5 | 51.9 | 51.9 | 59.6 | 62.6 | 65.3 | 59.4 | 68.5 |
| LR (tuned C) | 61.8 | 63.9 | 74.4 | 77.7 | 74.1 | 73.7 | 72.6 | 77.6 |

**LLaVA-1.5-7B, text-raw** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|---|
| CCS | 51.6 | 52.0 | 51.8 | 58.8 | 60.5 | 61.0 | 59.9 | 67.3 |
| LR (tuned C) | 62.1 | 64.6 | 74.3 | 77.4 | 73.6 | 73.4 | 72.8 | 76.8 |

**Qwen2-VL-7B, text-chat** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 51.5 | 52.6 | 57.2 | 75.9 | 90.0 | 89.5 | 87.0 |
| LR (tuned C) | 60.8 | 68.9 | 76.2 | 87.2 | 91.3 | 90.8 | 90.8 |

**Qwen2-VL-7B, text-raw** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 52.0 | 52.5 | 60.2 | 74.4 | 87.6 | 88.2 | 86.9 |
| LR (tuned C) | 61.1 | 68.1 | 75.9 | 86.0 | 89.7 | 89.8 | 89.8 |

**Qwen2.5-VL-7B, text-chat** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 51.9 | 51.5 | 52.4 | 77.4 | 88.3 | 89.8 | 89.8 |
| LR (tuned C) | 61.3 | 67.0 | 74.0 | 86.5 | 91.3 | 91.4 | 91.2 |

**Qwen2.5-VL-7B, text-raw** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 52.0 | 52.4 | 58.8 | 77.9 | 86.2 | 87.6 | 81.6 |
| LR (tuned C) | 60.8 | 67.8 | 75.0 | 86.6 | 90.6 | 90.8 | 90.5 |

**Qwen2-7B-Instruct, text-raw** (position final)

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| CCS | 52.5 | 51.6 | 58.8 | 80.4 | 87.5 | 88.5 | 86.7 |
| LR (tuned C) | 60.6 | 68.4 | 75.2 | 87.0 | 90.4 | 90.5 | 90.4 |
