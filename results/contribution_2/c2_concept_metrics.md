# C2 Concept Prediction Performance

Evaluation is performed on the independent Test split. Concept predictions use a fixed 0.5 sigmoid threshold. No threshold is tuned on Test.

Primary values are mean ± sample SD across seeds 42, 100, and 2026.

## Overall concept performance

| Model           | Macro Concept F1   | Macro Concept AUROC   |
|:----------------|:-------------------|:----------------------|
| B6_PureCBM      | 0.5171 ± 0.0633    | 0.7868 ± 0.0597       |
| Proposed_Hybrid | 0.5606 ± 0.0247    | 0.8437 ± 0.0136       |

## Per-seed results

| Model           |   Seed |   Macro Concept F1 |   Macro Concept AUROC |
|:----------------|-------:|-------------------:|----------------------:|
| B6_PureCBM      |     42 |             0.4440 |                0.7180 |
| B6_PureCBM      |    100 |             0.5508 |                0.8178 |
| B6_PureCBM      |   2026 |             0.5564 |                0.8245 |
| Proposed_Hybrid |     42 |             0.5704 |                0.8553 |
| Proposed_Hybrid |    100 |             0.5789 |                0.8472 |
| Proposed_Hybrid |   2026 |             0.5325 |                0.8286 |