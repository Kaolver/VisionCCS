Generated from the .out files in new_results.

**\*** = nonlinear result from the OLD code (lr 0.01, unseeded), pending re-run with the current code (lr 1e-3, seed 42). Unmarked nonlinear cells are from the current code.

Per-model averages are the values reported in the files. Mean columns and task-type tables are computed from the rounded per-category values, so they can differ by 0.1.

## VQA2

**Qwen2.5**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 76.5 | 74.7 | 77.9 |
| Linear supervised probe | 86.8 | 79.7 | 69.3 | 78.6 |
| Nonlinear supervised (MLP) | 83.7* | 79.6* | 72.6* | 78.7* |
| Linear CCS | 87.8 | 82.9 | 79.0 | 83.2 |
| Nonlinear CCS (MLP) | 87.6* | 81.8* | 78.7* | 82.7* |

**LLaVA**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 79.0 | 74.0 | 71.9 | 75.0 |
| Linear supervised probe | 84.1 | 76.6 | 72.9 | 77.9 |
| Nonlinear supervised (MLP) | 83.7* | 74.6* | 67.0* | 75.1* |
| Linear CCS | 51.6 | 74.1 | 60.4 | 62.0 |
| Nonlinear CCS (MLP) | 84.1* | 67.0* | 58.0* | 69.7* |

**Qwen2**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 77.8 | 78.3 | 79.5 |
| Linear supervised probe | 86.4 | 82.6 | 78.5 | 82.5 |
| Nonlinear supervised (MLP) | 88.3* | 82.3* | 76.7* | 82.4* |
| Linear CCS | 89.9 | 82.0 | 74.7 | 82.2 |
| Nonlinear CCS (MLP) | 51.8* | 81.4* | 77.9* | 70.3* |

**Cross-model averages (VQA2)**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 77.9 | 75.0 | 79.5 | 77.5 |
| Linear supervised probe | 78.6 | 77.9 | 82.5 | 79.7 |
| Nonlinear supervised (MLP) | 78.7* | 75.1* | 82.4* | 78.7 |
| Linear CCS | 83.2 | 62.0 | 82.2 | 75.8 |
| Nonlinear CCS (MLP) | 82.7* | 69.7* | 70.3* | 74.2 |

**Per task type (VQA2, mean over models)**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object detection | 81.3 | 85.8 | 85.2 | 76.4 | 74.5 |
| Attribute recognition | 76.1 | 79.6 | 78.8 | 79.7 | 76.7 |
| Spatial recognition | 75.0 | 73.6 | 72.1 | 71.4 | 71.5 |
| Mean | 77.5 | 79.7 | 78.7 | 75.8 | 74.2 |

## VG

**Qwen2.5**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 81.8 | 84.0 | 81.4 | 82.4 |
| Linear supervised probe | 70.0 | 84.4 | 79.5 | 78.0 |
| Nonlinear supervised (MLP) | 85.1 | 86.6 | 83.6 | 85.1 |
| Linear CCS | 76.3 | 83.7 | 71.3 | 77.1 |
| Nonlinear CCS (MLP) | 71.8 | 83.4 | 73.8 | 76.3 |

**LLaVA**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 84.2 | 79.9 | 81.9 | 82.0 |
| Linear supervised probe | 80.7 | 80.5 | 79.2 | 80.1 |
| Nonlinear supervised (MLP) | 82.5 | 81.6 | 83.2 | 82.4 |
| Linear CCS | 51.8 | 71.8 | 56.6 | 60.0 |
| Nonlinear CCS (MLP) | 53.4 | 71.6 | 58.2 | 61.1 |

**Qwen2**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 80.7 | 80.1 | 76.5 | 79.1 |
| Linear supervised probe | 83.9 | 85.8 | 83.7 | 84.5 |
| Nonlinear supervised (MLP) | 85.3 | 85.1 | 85.1 | 85.2 |
| Linear CCS | 75.6 | 78.0 | 50.4 | 68.0 |
| Nonlinear CCS (MLP) | 75.8 | 78.4 | 50.9 | 68.4 |

**Cross-model averages (VG)**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 82.4 | 82.0 | 79.1 | 81.2 |
| Linear supervised probe | 78.0 | 80.1 | 84.5 | 80.9 |
| Nonlinear supervised (MLP) | 85.1 | 82.4 | 85.2 | 84.2 |
| Linear CCS | 77.1 | 60.0 | 68.0 | 68.4 |
| Nonlinear CCS (MLP) | 76.3 | 61.1 | 68.4 | 68.6 |

**Per task type (VG, mean over models)**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object | 82.2 | 78.2 | 84.3 | 67.9 | 67.0 |
| Attribute | 81.3 | 83.6 | 84.4 | 77.8 | 77.8 |
| Spatial | 79.9 | 80.8 | 84.0 | 59.4 | 61.0 |
| Mean | 81.2 | 80.9 | 84.2 | 68.4 | 68.6 |

## Combined (VQA2 + VG)

**Per model, averaged over both datasets**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 80.2 | 78.5 | 79.3 | 79.3 |
| Linear supervised probe | 78.3 | 79.0 | 83.5 | 80.3 |
| Nonlinear supervised (MLP) | 81.9 | 78.8 | 83.8 | 81.5 |
| Linear CCS | 80.2 | 61.0 | 75.1 | 72.1 |
| Nonlinear CCS (MLP) | 79.5 | 65.4 | 69.3 | 71.4 |

**Per dataset, averaged over models**

| Method | VQA2 | VG | Mean |
|---|---|---|---|
| Logistic regression | 77.5 | 81.2 | 79.3 |
| Linear supervised probe | 79.7 | 80.9 | 80.3 |
| Nonlinear supervised (MLP) | 78.7 | 84.2 | 81.5 |
| Linear CCS | 75.8 | 68.4 | 72.1 |
| Nonlinear CCS (MLP) | 74.2 | 68.6 | 71.4 |

**Per task type, averaged over models and datasets**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object | 81.8 | 82.0 | 84.8 | 72.2 | 70.8 |
| Attribute | 78.7 | 81.6 | 81.6 | 78.8 | 77.3 |
| Spatial | 77.5 | 77.2 | 78.0 | 65.4 | 66.2 |
| Mean | 79.3 | 80.3 | 81.5 | 72.1 | 71.4 |
