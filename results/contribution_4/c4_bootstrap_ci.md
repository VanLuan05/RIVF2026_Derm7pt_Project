# Bootstrap Confidence Intervals — Proposed Hybrid

Stratified percentile bootstrap with 1000 replicates. Resampling is performed within each disease class, so every replicate preserves the Test-set class counts.

The paper's primary across-training-run uncertainty remains mean ± sample SD over seeds. The probability ensemble is a secondary analysis and should be labeled as such.

| Analysis                                | Metric            |   Point Estimate |   95% CI Lower |   95% CI Upper |
|:----------------------------------------|:------------------|-----------------:|---------------:|---------------:|
| Seed 42                                 | Accuracy          |           0.6974 |         0.6250 |         0.7697 |
| Seed 42                                 | Balanced Accuracy |           0.5394 |         0.4167 |         0.6778 |
| Seed 42                                 | Macro F1          |           0.5076 |         0.4022 |         0.6246 |
| Seed 100                                | Accuracy          |           0.7039 |         0.6447 |         0.7633 |
| Seed 100                                | Balanced Accuracy |           0.4865 |         0.3881 |         0.6022 |
| Seed 100                                | Macro F1          |           0.4746 |         0.3819 |         0.5780 |
| Seed 2026                               | Accuracy          |           0.6513 |         0.5722 |         0.7237 |
| Seed 2026                               | Balanced Accuracy |           0.4993 |         0.3788 |         0.6131 |
| Seed 2026                               | Macro F1          |           0.4521 |         0.3589 |         0.5435 |
| 3-seed probability ensemble (secondary) | Accuracy          |           0.7237 |         0.6579 |         0.7895 |
| 3-seed probability ensemble (secondary) | Balanced Accuracy |           0.5344 |         0.4187 |         0.6453 |
| 3-seed probability ensemble (secondary) | Macro F1          |           0.5141 |         0.4041 |         0.6130 |