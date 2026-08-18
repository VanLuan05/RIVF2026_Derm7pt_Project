# Contribution 1 - Cross-Attention Evaluation

Independent Test set evaluation across seeds 42, 100, and 2026.

| Model | Fusion | Macro-F1 | Balanced Accuracy | AUROC |
|---|---|---:|---:|---:|
| B5_Dual_Metadata | Concatenation | 0.5334 ± 0.0213 | 0.5779 ± 0.0376 | 0.8852 ± 0.0103 |
| C1_CrossAttention | Cross-Attention | 0.5410 ± 0.0241 | 0.5699 ± 0.0482 | 0.8263 ± 0.0226 |
