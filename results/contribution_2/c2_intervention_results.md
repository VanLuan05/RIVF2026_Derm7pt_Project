# C2 Oracle Concept Intervention

Ground-truth Derm7pt concept labels are substituted as soft oracle probabilities (0.05/0.95). This is an oracle diagnostic analysis, not a prospective clinician intervention study.

## Summary

| Model           | Macro F1 AI     | Macro F1 Oracle   |   Mean Delta Macro F1 | Balanced Accuracy AI   | Balanced Accuracy Oracle   |   Mean Delta Balanced Accuracy |
|:----------------|:----------------|:------------------|----------------------:|:-----------------------|:---------------------------|-------------------------------:|
| B6_PureCBM      | 0.4018 ± 0.0847 | 0.2281 ± 0.0743   |               -0.1737 | 0.4665 ± 0.0706        | 0.2476 ± 0.0129            |                        -0.2189 |
| Proposed_Hybrid | 0.4489 ± 0.0287 | 0.4472 ± 0.0268   |               -0.0017 | 0.4779 ± 0.0230        | 0.4771 ± 0.0223            |                        -0.0007 |

## Per-seed results

| Model           |   Seed |   Accuracy AI |   Accuracy Oracle |   Delta Accuracy |   Balanced Accuracy AI |   Balanced Accuracy Oracle |   Delta Balanced Accuracy |   Macro F1 AI |   Macro F1 Oracle |   Delta Macro F1 |
|:----------------|-------:|--------------:|------------------:|-----------------:|-----------------------:|---------------------------:|--------------------------:|--------------:|------------------:|-----------------:|
| B6_PureCBM      |     42 |        0.4079 |            0.1711 |          -0.2368 |                 0.3927 |                     0.2332 |                   -0.1595 |        0.3072 |            0.1424 |          -0.1648 |
| B6_PureCBM      |    100 |        0.6711 |            0.5395 |          -0.1316 |                 0.5335 |                     0.2519 |                   -0.2816 |        0.4705 |            0.2722 |          -0.1983 |
| B6_PureCBM      |   2026 |        0.6250 |            0.5066 |          -0.1184 |                 0.4734 |                     0.2578 |                   -0.2156 |        0.4278 |            0.2698 |          -0.1580 |
| Proposed_Hybrid |     42 |        0.6711 |            0.6645 |          -0.0066 |                 0.4924 |                     0.4902 |                   -0.0022 |        0.4707 |            0.4655 |          -0.0052 |
| Proposed_Hybrid |    100 |        0.6842 |            0.6842 |           0.0000 |                 0.4899 |                     0.4899 |                    0.0000 |        0.4596 |            0.4596 |           0.0000 |
| Proposed_Hybrid |   2026 |        0.6316 |            0.6316 |           0.0000 |                 0.4513 |                     0.4513 |                    0.0000 |        0.4164 |            0.4164 |           0.0000 |