# C2 Disease Classification on Independent Test Set

Evaluation uses the corrected C2 checkpoints for B6_PureCBM and Proposed_Hybrid across the three locked seeds. The Test set is used only for final evaluation.

## Summary

| Model           | Accuracy        | Balanced Accuracy   | Macro F1        | Macro Precision   | Macro Recall    | Macro Specificity   | One-vs-Rest AUROC   |
|:----------------|:----------------|:--------------------|:----------------|:------------------|:----------------|:--------------------|:--------------------|
| B6_PureCBM      | 0.5680 ± 0.1405 | 0.4665 ± 0.0706     | 0.4018 ± 0.0847 | 0.4180 ± 0.0567   | 0.4665 ± 0.0706 | 0.8942 ± 0.0272     | 0.7495 ± 0.0197     |
| Proposed_Hybrid | 0.6623 ± 0.0274 | 0.4779 ± 0.0230     | 0.4489 ± 0.0287 | 0.4434 ± 0.0179   | 0.4779 ± 0.0230 | 0.9070 ± 0.0073     | 0.8465 ± 0.0238     |

## Per-seed results

| Model           |   Seed |   Accuracy |   Balanced Accuracy |   Macro F1 |   Macro Precision |   Macro Recall |   Macro Specificity |   One-vs-Rest AUROC |
|:----------------|-------:|-----------:|--------------------:|-----------:|------------------:|---------------:|--------------------:|--------------------:|
| B6_PureCBM      |     42 |     0.4079 |              0.3927 |     0.3072 |            0.3567 |         0.3927 |              0.8632 |              0.7269 |
| B6_PureCBM      |    100 |     0.6711 |              0.5335 |     0.4705 |            0.4686 |         0.5335 |              0.9141 |              0.7631 |
| B6_PureCBM      |   2026 |     0.6250 |              0.4734 |     0.4278 |            0.4286 |         0.4734 |              0.9053 |              0.7585 |
| Proposed_Hybrid |     42 |     0.6711 |              0.4924 |     0.4707 |            0.4574 |         0.4924 |              0.8995 |              0.8432 |
| Proposed_Hybrid |    100 |     0.6842 |              0.4899 |     0.4596 |            0.4495 |         0.4899 |              0.9140 |              0.8717 |
| Proposed_Hybrid |   2026 |     0.6316 |              0.4513 |     0.4164 |            0.4232 |         0.4513 |              0.9074 |              0.8245 |