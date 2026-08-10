# Sequential CBM Concept-Quality Gap Analysis

Two downstream classifiers are trained using Train only: one on predicted concept probabilities and one on soft ground-truth concepts (oracle upper bound). Test is used only for final evaluation. This is an oracle analysis, not a prospective clinician study.

## Per-seed results

|      Seed |   Accuracy Predicted Concepts |   Accuracy Oracle Concepts |   Macro F1 Predicted Concepts |   Macro F1 Oracle Concepts |   Oracle Gap Macro F1 |
|----------:|------------------------------:|---------------------------:|------------------------------:|---------------------------:|----------------------:|
|   42.0000 |                        0.7763 |                     0.5197 |                        0.5274 |                     0.4237 |               -0.1038 |
|  100.0000 |                        0.7303 |                     0.5197 |                        0.4804 |                     0.4237 |               -0.0568 |
| 2026.0000 |                        0.7368 |                     0.5197 |                        0.6030 |                     0.4237 |               -0.1794 |

## Summary

| Seed             | Accuracy Predicted Concepts   | Accuracy Oracle Concepts   | Macro F1 Predicted Concepts   | Macro F1 Oracle Concepts   |   Oracle Gap Macro F1 |
|:-----------------|:------------------------------|:---------------------------|:------------------------------|:---------------------------|----------------------:|
| Mean ± sample SD | 0.7478 ± 0.0249               | 0.5197 ± 0.0000            | 0.5370 ± 0.0619               | 0.4237 ± 0.0000            |               -0.1133 |