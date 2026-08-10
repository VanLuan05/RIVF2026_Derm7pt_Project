# Oracle Concept Intervention Analysis

Ground-truth Derm7pt concept labels are substituted as soft oracle probabilities (0.05/0.95). This diagnoses concept dependence and must not be described as a real doctor intervention study.

## Per-seed results

| Model           |   Seed |   Accuracy AI |   Accuracy Oracle |   Delta Accuracy |   Macro F1 AI |   Macro F1 Oracle |   Delta Macro F1 |
|:----------------|-------:|--------------:|------------------:|-----------------:|--------------:|------------------:|-----------------:|
| B6_PureCBM      |     42 |        0.6250 |            0.5132 |          -0.1118 |        0.4246 |            0.2831 |          -0.1415 |
| B6_PureCBM      |    100 |        0.6250 |            0.5592 |          -0.0658 |        0.4341 |            0.2728 |          -0.1613 |
| B6_PureCBM      |   2026 |        0.6579 |            0.3224 |          -0.3355 |        0.5340 |            0.2209 |          -0.3132 |
| Proposed_Hybrid |     42 |        0.6974 |            0.6974 |           0.0000 |        0.5076 |            0.5076 |           0.0000 |
| Proposed_Hybrid |    100 |        0.7039 |            0.7039 |           0.0000 |        0.4746 |            0.4746 |           0.0000 |
| Proposed_Hybrid |   2026 |        0.6513 |            0.6513 |           0.0000 |        0.4521 |            0.4521 |           0.0000 |

## Summary

| Model           | Accuracy AI     | Accuracy Oracle   | Macro F1 AI     | Macro F1 Oracle   |   Mean Delta Macro F1 |
|:----------------|:----------------|:------------------|:----------------|:------------------|----------------------:|
| B6_PureCBM      | 0.6360 ± 0.0190 | 0.4649 ± 0.1256   | 0.4642 ± 0.0606 | 0.2589 ± 0.0334   |               -0.2053 |
| Proposed_Hybrid | 0.6842 ± 0.0287 | 0.6842 ± 0.0287   | 0.4781 ± 0.0279 | 0.4781 ± 0.0279   |                0      |