Generated from the .out files in new_results.

All nonlinear results are from the current code (lr 1e-3, seed 42); linear results use lr 0.01, unseeded.

Per-model averages are the values reported in the files; every Mean row and column averages those. The per-task rows average the rounded per-category values, so they can differ by 0.1 from the Mean row.

## VQA2

**Qwen2.5**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 76.5 | 74.7 | 77.9 |
| Linear supervised probe | 86.8 | 79.7 | 69.3 | 78.6 |
| Nonlinear supervised (MLP) | 87.6 | 81.4 | 77.0 | 82.0 |
| Linear CCS | 87.8 | 82.9 | 79.0 | 83.2 |
| Nonlinear CCS (MLP) | 88.0 | 82.3 | 79.0 | 83.1 |

**LLaVA**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 79.0 | 74.0 | 71.9 | 75.0 |
| Linear supervised probe | 84.1 | 76.6 | 72.9 | 77.9 |
| Nonlinear supervised (MLP) | 83.7 | 77.2 | 73.1 | 78.0 |
| Linear CCS | 51.6 | 74.1 | 60.4 | 62.0 |
| Nonlinear CCS (MLP) | 73.4 | 76.1 | 68.8 | 72.8 |

**Qwen2**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 77.8 | 78.3 | 79.5 |
| Linear supervised probe | 86.4 | 82.6 | 78.5 | 82.5 |
| Nonlinear supervised (MLP) | 87.8 | 83.0 | 77.0 | 82.6 |
| Linear CCS | 89.9 | 82.0 | 74.7 | 82.2 |
| Nonlinear CCS (MLP) | 88.5 | 82.6 | 79.8 | 83.6 |

**Cross-model averages (VQA2)**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 77.9 | 75.0 | 79.5 | 77.5 |
| Linear supervised probe | 78.6 | 77.9 | 82.5 | 79.7 |
| Nonlinear supervised (MLP) | 82.0 | 78.0 | 82.6 | 80.9 |
| Linear CCS | 83.2 | 62.0 | 82.2 | 75.8 |
| Nonlinear CCS (MLP) | 83.1 | 72.8 | 83.6 | 79.8 |

**Per task type (VQA2, mean over models)**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object detection | 81.3 | 85.8 | 86.4 | 76.4 | 83.3 |
| Attribute recognition | 76.1 | 79.6 | 80.5 | 79.7 | 80.3 |
| Spatial recognition | 75.0 | 73.6 | 75.7 | 71.4 | 75.9 |
| Mean | 77.5 | 79.7 | 80.9 | 75.8 | 79.8 |

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
| Nonlinear supervised (MLP) | 83.5 | 80.2 | 83.9 | 82.5 |
| Linear CCS | 80.2 | 61.0 | 75.1 | 72.1 |
| Nonlinear CCS (MLP) | 79.7 | 67.0 | 76.0 | 74.2 |

**Per dataset, averaged over models**

| Method | VQA2 | VG | Mean |
|---|---|---|---|
| Logistic regression | 77.5 | 81.2 | 79.3 |
| Linear supervised probe | 79.7 | 80.9 | 80.3 |
| Nonlinear supervised (MLP) | 80.9 | 84.2 | 82.5 |
| Linear CCS | 75.8 | 68.4 | 72.1 |
| Nonlinear CCS (MLP) | 79.8 | 68.6 | 74.2 |

**Per task type, averaged over models and datasets**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object | 81.8 | 82.0 | 85.3 | 72.2 | 75.2 |
| Attribute | 78.7 | 81.6 | 82.5 | 78.8 | 79.1 |
| Spatial | 77.5 | 77.2 | 79.8 | 65.4 | 68.4 |
| Mean | 79.3 | 80.3 | 82.5 | 72.1 | 74.2 |
