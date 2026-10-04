**† marks a failed training run, not a real result.** These are LLaVA VG nonlinear supervised (all three categories) and Qwen2.5 VG nonlinear supervised on attribute. That second one is a new finding: train accuracy is 50.8% and it almost always predicts "no", so it's the same collapse as the LLaVA run. Every average that includes these cells is pulled down.

Per-model averages are the values reported in the files. The Mean columns and task-type tables are computed from the rounded per-category values, so they can be off by 0.1.

## VQA2

**Qwen2.5**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 76.5 | 74.7 | 77.9 |
| Linear supervised probe | 86.8 | 79.7 | 69.3 | 78.6 |
| Nonlinear supervised (MLP) | 83.7 | 79.6 | 72.6 | 78.7 |
| Linear CCS | 87.8 | 82.9 | 79.0 | 83.2 |
| Nonlinear CCS (MLP) | 87.6 | 81.8 | 78.7 | 82.7 |

**LLaVA**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 79.0 | 74.0 | 71.9 | 75.0 |
| Linear supervised probe | 84.1 | 76.6 | 72.9 | 77.9 |
| Nonlinear supervised (MLP) | 83.7 | 74.6 | 67.0 | 75.1 |
| Linear CCS | 51.6 | 74.1 | 60.4 | 62.0 |
| Nonlinear CCS (MLP) | 84.1 | 67.0 | 58.0 | 69.7 |

**Qwen2**

| Method | Object det. | Attribute rec. | Spatial rec. | Average |
|---|---|---|---|---|
| Logistic regression | 82.4 | 77.8 | 78.3 | 79.5 |
| Linear supervised probe | 86.4 | 82.6 | 78.5 | 82.5 |
| Nonlinear supervised (MLP) | 88.3 | 82.3 | 76.7 | 82.4 |
| Linear CCS | 89.9 | 82.0 | 74.7 | 82.2 |
| Nonlinear CCS (MLP) | 51.8 | 81.4 | 77.9 | 70.3 |

**Cross-model averages (VQA2)**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 77.9 | 75.0 | 79.5 | 77.4 |
| Linear supervised probe | 78.6 | 77.9 | 82.5 | 79.7 |
| Nonlinear supervised (MLP) | 78.7 | 75.1 | 82.4 | 78.7 |
| Linear CCS | 83.2 | 62.0 | 82.2 | 75.8 |
| Nonlinear CCS (MLP) | 82.7 | 69.7 | 70.3 | 74.3 |

**Per task type (VQA2, mean over models)**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object detection | 81.3 | 85.8 | 85.2 | 76.4 | 74.5 |
| Attribute recognition | 76.1 | 79.6 | 78.8 | 79.7 | 76.7 |
| Spatial recognition | 75.0 | 73.6 | 72.1 | 71.4 | 71.5 |
| Mean | 77.4 | 79.7 | 78.7 | 75.8 | 74.3 |

## VG

**Qwen2.5**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 81.8 | 84.0 | 81.4 | 82.4 |
| Linear supervised probe | 70.0 | 84.4 | 79.5 | 78.0 |
| Nonlinear supervised (MLP) | 78.1 | 51.9† | 83.7 | 71.2 |
| Linear CCS | 76.3 | 83.7 | 71.3 | 77.1 |
| Nonlinear CCS (MLP) | 75.4 | 83.0 | 73.8 | 77.4 |

**LLaVA**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 84.2 | 79.9 | 81.9 | 82.0 |
| Linear supervised probe | 80.7 | 80.5 | 79.2 | 80.1 |
| Nonlinear supervised (MLP) | 50.7† | 50.7† | 50.7† | 50.7† |
| Linear CCS | 51.8 | 71.8 | 56.6 | 60.0 |
| Nonlinear CCS (MLP) | 51.2 | 67.3 | 65.3 | 61.2 |

**Qwen2**

| Method | Object | Attribute | Spatial | Average |
|---|---|---|---|---|
| Logistic regression | 80.7 | 80.1 | 76.5 | 79.1 |
| Linear supervised probe | 83.9 | 85.8 | 83.7 | 84.5 |
| Nonlinear supervised (MLP) | 84.7 | 73.6 | 84.8 | 81.0 |
| Linear CCS | 75.6 | 78.0 | 50.4 | 68.0 |
| Nonlinear CCS (MLP) | 76.3 | 79.1 | 68.1 | 74.5 |

**Cross-model averages (VG)**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 82.4 | 82.0 | 79.1 | 81.2 |
| Linear supervised probe | 78.0 | 80.1 | 84.5 | 80.9 |
| Nonlinear supervised (MLP) | 71.2† | 50.7† | 81.0 | 67.7† |
| Linear CCS | 77.1 | 60.0 | 68.0 | 68.4 |
| Nonlinear CCS (MLP) | 77.4 | 61.2 | 74.5 | 71.1 |

**Per task type (VG, mean over models)**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object | 82.2 | 78.2 | 71.2† | 67.9 | 67.6 |
| Attribute | 81.3 | 83.6 | 58.7† | 77.8 | 76.5 |
| Spatial | 79.9 | 80.8 | 73.1† | 59.4 | 69.1 |
| Mean | 81.2 | 80.9 | 67.7† | 68.4 | 71.1 |

## Combined (VQA2 + VG)

**Per model, averaged over both datasets**

| Method | Qwen2.5 | LLaVA | Qwen2 | Mean |
|---|---|---|---|---|
| Logistic regression | 80.1 | 78.5 | 79.3 | 79.3 |
| Linear supervised probe | 78.3 | 79.0 | 83.5 | 80.3 |
| Nonlinear supervised (MLP) | 74.9† | 62.9† | 81.7 | 73.2† |
| Linear CCS | 80.2 | 61.0 | 75.1 | 72.1 |
| Nonlinear CCS (MLP) | 80.1 | 65.5 | 72.4 | 72.7 |

**Per dataset, averaged over models**

| Method | VQA2 | VG | Mean |
|---|---|---|---|
| Logistic regression | 77.4 | 81.2 | 79.3 |
| Linear supervised probe | 79.7 | 80.9 | 80.3 |
| Nonlinear supervised (MLP) | 78.7 | 67.7† | 73.2† |
| Linear CCS | 75.8 | 68.4 | 72.1 |
| Nonlinear CCS (MLP) | 74.3 | 71.1 | 72.7 |

**Per task type, averaged over models and datasets**

| Task type | LogReg | Linear sup. | Nonlinear sup. | Linear CCS | Nonlinear CCS |
|---|---|---|---|---|---|
| Object | 81.8 | 82.0 | 78.2† | 72.2 | 71.1 |
| Attribute | 78.7 | 81.6 | 68.8† | 78.8 | 76.6 |
| Spatial | 77.5 | 77.2 | 72.6† | 65.4 | 70.3 |
| Mean | 79.3 | 80.3 | 73.2† | 72.1 | 72.7 |

## What stands out

- **On VG, CCS falls behind the supervised baselines.** On VQA2, linear CCS is the best method on both Qwen models. On VG, the supervised probes win on every model, and CCS averages 68–71 against 81 for the supervised probes.
- **CCS is near chance in five cells:**
  - LLaVA object: linear CCS 51.6 on VQA2 and 51.8 on VG; nonlinear CCS 51.2 on VG.
  - Qwen2 VG spatial, linear CCS: 50.4.
  - Qwen2 VQA2 object, nonlinear CCS: 51.8.

  The logistic-regression sanity check reaches 77–84% on the same hidden states, so the information is there and CCS just doesn't find it. The misfiled Qwen2 VQA2 re-run (`VQ/Qwen2/nonlinear-ccs_qwen2_VG_2.out`) also lands at 51.6 on object, so that cell looks like a consistent failure rather than bad luck with the seed.
- **Some per-cell differences are large.** For example, Qwen2.5 linear supervised object is 70.0 on VG versus 86.8 on VQA2. The linear CCS and linear supervised runs are still unseeded, so treat single-run differences of a few points with caution.

The nonlinear supervised column will change once step 2 is fixed and the nonlinear jobs are re-run, and the † cells should be replaced then.