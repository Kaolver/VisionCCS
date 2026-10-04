### Prompt matched to CCS (no "Answer yes or no.")

| dataset | model | object | attribute | spatial | all | says yes | n | skipped |
|---|---|---|---|---|---|---|---|---|
| VQAv2 | LLaVA-1.5-7B | 83.3 (78.9) | 76.5 (72.7) | 75.6 (70.0) | 77.9 (73.7) | 66.7% | 5648 | 0 |
| VQAv2 | Qwen2-VL-7B | 85.6 (85.6) | 84.0 (83.5) | 79.5 (80.1) | 83.6 (83.4) | 54.6% | 5648 | 0 |
| VQAv2 | Qwen2.5-VL-7B | 85.9 (83.3) | 83.5 (80.3) | 80.1 (74.8) | 83.5 (80.0) | 35.6% | 5648 | 0 |
| Visual Genome | LLaVA-1.5-7B | 72.2 (68.1) | 75.8 (75.5) | 71.3 (57.7) | 73.1 (67.1) | 76.0% | 36000 | 0 |
| Visual Genome | Qwen2-VL-7B | 74.2 (73.8) | 82.9 (82.9) | 77.3 (72.3) | 78.1 (76.3) | 61.1% | 36000 | 0 |
| Visual Genome | Qwen2.5-VL-7B | 73.7 (74.4) | 84.0 (75.3) | 77.9 (69.5) | 78.5 (73.1) | 35.5% | 36000 | 0 |

### With "Answer yes or no." instruction

| dataset | model | object | attribute | spatial | all | says yes | n | skipped |
|---|---|---|---|---|---|---|---|---|
| VQAv2 | LLaVA-1.5-7B | 83.5 (83.5) | 77.7 (77.8) | 77.3 (77.3) | 78.9 (79.1) | 49.6% | 5648 | 0 |
| VQAv2 | Qwen2-VL-7B | 85.8 (85.8) | 84.2 (84.1) | 79.5 (80.3) | 83.7 (83.9) | 49.2% | 5648 | 0 |
| VQAv2 | Qwen2.5-VL-7B | 86.7 (87.3) | 84.5 (83.8) | 81.4 (79.9) | 84.5 (84.0) | 46.1% | 5648 | 0 |
| Visual Genome | LLaVA-1.5-7B | 72.4 (72.5) | 76.1 (75.9) | 68.1 (64.7) | 72.2 (71.0) | 59.9% | 36000 | 0 |
| Visual Genome | Qwen2-VL-7B | 75.2 (75.8) | 82.8 (81.7) | 78.8 (78.2) | 78.9 (78.6) | 51.0% | 36000 | 0 |
| Visual Genome | Qwen2.5-VL-7B | 75.0 (76.3) | 84.7 (82.4) | 79.3 (79.2) | 79.7 (79.3) | 48.8% | 36000 | 0 |

Cells are calibrated accuracy % (raw accuracy %). Raw: the model's own Yes/No preference (yes-logit > no-logit). Calibrated (Burns et al. 2022): the half of items with the largest yes-no margin are predicted yes, which removes the model's yes/no bias. "says yes" is how often the uncalibrated model answers yes; the data are 50/50.
