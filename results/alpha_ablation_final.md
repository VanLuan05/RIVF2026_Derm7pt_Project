# Final Alpha Ablation on Validation Set

Alpha is selected using mean Validation Disease Macro-F1 across three seeds.
Concept Macro-F1 is used as a secondary criterion.
The Test set is not used for selection.

| Alpha | Seed 42 Disease F1 | Seed 100 Disease F1 | Seed 2026 Disease F1 | Mean Disease F1 | SD Disease F1 | Mean Concept F1 | SD Concept F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.5278 | 0.5742 | 0.5668 | 0.5562 | 0.0249 | 0.4630 | 0.0299 |
| 1.0 | 0.5299 | 0.5375 | 0.5877 | 0.5517 | 0.0314 | 0.4681 | 0.0049 |
| 2.0 | 0.5632 | 0.5574 | 0.5829 | **0.5678** | 0.0133 | **0.4832** | 0.0143 |
| 3.0 | 0.5198 | 0.5671 | 0.5808 | 0.5559 | 0.0320 | 0.4656 | 0.0176 |

**Selected alpha: 2.0**

> Restored from the previously completed C2 alpha-tuning run after moving to a new Colab account.
