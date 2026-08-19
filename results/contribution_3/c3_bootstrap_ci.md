# Bootstrap Confidence Intervals — Proposed Hybrid

Stratified percentile bootstrap with 1000 replicates. Resampling is performed within each disease class, so every replicate preserves the Test-set class counts.

The paper's primary across-training-run uncertainty remains mean ± sample SD over seeds. The probability ensemble is a secondary analysis and should be labeled as such.

| Analysis                                | Metric            |   Point Estimate |   95% CI Lower |   95% CI Upper |
|:----------------------------------------|:------------------|-----------------:|---------------:|---------------:|
| Seed 42                                 | Accuracy          |           0.6711 |         0.5987 |         0.7368 |
| Seed 42                                 | Balanced Accuracy |           0.4924 |         0.3795 |         0.6250 |
| Seed 42                                 | Macro F1          |           0.4707 |         0.3681 |         0.5857 |
| Seed 100                                | Accuracy          |           0.6842 |         0.6184 |         0.7500 |
| Seed 100                                | Balanced Accuracy |           0.4899 |         0.3959 |         0.6087 |
| Seed 100                                | Macro F1          |           0.4596 |         0.3761 |         0.5652 |
| Seed 2026                               | Accuracy          |           0.6316 |         0.5526 |         0.7041 |
| Seed 2026                               | Balanced Accuracy |           0.4513 |         0.3452 |         0.5755 |
| Seed 2026                               | Macro F1          |           0.4164 |         0.3347 |         0.5080 |
| 3-seed probability ensemble (secondary) | Accuracy          |           0.6645 |         0.5921 |         0.7303 |
| 3-seed probability ensemble (secondary) | Balanced Accuracy |           0.4627 |         0.3691 |         0.5691 |
| 3-seed probability ensemble (secondary) | Macro F1          |           0.4383 |         0.3577 |         0.5279 |