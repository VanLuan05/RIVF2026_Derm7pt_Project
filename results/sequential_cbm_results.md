# Sequential CBM Concept-Quality Gap Analysis

Two downstream classifiers are trained using Train only: one on predicted concept probabilities and one on soft ground-truth concepts (oracle upper bound). Test is used only for final evaluation. This is an oracle analysis, not a prospective clinician study.

## Per-seed results

|      Seed |   Accuracy Predicted Concepts |   Accuracy Oracle Concepts |   Macro F1 Predicted Concepts |   Macro F1 Oracle Concepts |   Oracle Gap Macro F1 |
|----------:|------------------------------:|---------------------------:|------------------------------:|---------------------------:|----------------------:|
|   42.0000 |                        0.6053 |                     0.3947 |                        0.4420 |                     0.3725 |               -0.0695 |
|  100.0000 |                        0.7039 |                     0.3947 |                        0.4420 |                     0.3725 |               -0.0695 |
| 2026.0000 |                        0.7039 |                     0.3947 |                        0.4514 |                     0.3725 |               -0.0789 |

## Summary

| Seed             | Accuracy Predicted Concepts   | Accuracy Oracle Concepts   | Macro F1 Predicted Concepts   | Macro F1 Oracle Concepts   |   Oracle Gap Macro F1 |
|:-----------------|:------------------------------|:---------------------------|:------------------------------|:---------------------------|----------------------:|
| Mean ± sample SD | 0.6711 ± 0.0570               | 0.3947 ± 0.0000            | 0.4451 ± 0.0054               | 0.3725 ± 0.0000            |               -0.0726 |