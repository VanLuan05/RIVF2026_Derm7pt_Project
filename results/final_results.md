# Final Test Results

All model/hyperparameter selection is completed on Validation. The independent Test split is used only for final evaluation. Values are mean ± sample SD across the three locked seeds.

Selected concept-loss alpha: **2.0**.

## Overall metrics

| Model            | Seeds       | Accuracy        | Balanced Accuracy   | Macro F1        | Macro Precision   | Macro Recall    | Macro Specificity   | One-vs-Rest AUROC   |
|:-----------------|:------------|:----------------|:--------------------|:----------------|:------------------|:----------------|:--------------------|:--------------------|
| B1_Clinical_Only | 42,100,2026 | 0.6272 ± 0.0297 | 0.4591 ± 0.0302     | 0.4197 ± 0.0237 | 0.4072 ± 0.0234   | 0.4591 ± 0.0302 | 0.8962 ± 0.0035     | 0.7684 ± 0.0152     |
| B2_Derm_Only     | 42,100,2026 | 0.6974 ± 0.0132 | 0.5751 ± 0.0243     | 0.5253 ± 0.0234 | 0.5048 ± 0.0226   | 0.5751 ± 0.0243 | 0.9156 ± 0.0055     | 0.8672 ± 0.0212     |
| B3_Meta_Only     | 42,100,2026 | 0.4518 ± 0.0622 | 0.5120 ± 0.0333     | 0.3395 ± 0.0684 | 0.3540 ± 0.0742   | 0.5120 ± 0.0333 | 0.8585 ± 0.0130     | 0.7374 ± 0.0072     |
| B4_Dual_NoMeta   | 42,100,2026 | 0.6294 ± 0.0666 | 0.5218 ± 0.0567     | 0.4591 ± 0.0307 | 0.4522 ± 0.0213   | 0.5218 ± 0.0567 | 0.9039 ± 0.0136     | 0.8417 ± 0.0195     |
| B5_Dual_Metadata | 42,100,2026 | 0.7105 ± 0.0174 | 0.5779 ± 0.0376     | 0.5334 ± 0.0213 | 0.5129 ± 0.0156   | 0.5779 ± 0.0376 | 0.9220 ± 0.0054     | 0.8852 ± 0.0103     |
| B6_PureCBM       | 42,100,2026 | 0.6360 ± 0.0190 | 0.5320 ± 0.0633     | 0.4642 ± 0.0606 | 0.4514 ± 0.0567   | 0.5320 ± 0.0633 | 0.9045 ± 0.0049     | 0.8194 ± 0.0281     |
| Proposed_Hybrid  | 42,100,2026 | 0.6842 ± 0.0287 | 0.5084 ± 0.0276     | 0.4781 ± 0.0279 | 0.4660 ± 0.0250   | 0.5084 ± 0.0276 | 0.9133 ± 0.0072     | 0.8555 ± 0.0217     |

## Per-seed metrics

| Model            |   Seed |   Accuracy |   Balanced Accuracy |   Macro F1 |   Macro Precision |   Macro Recall |   Macro Specificity |   One-vs-Rest AUROC |
|:-----------------|-------:|-----------:|--------------------:|-----------:|------------------:|---------------:|--------------------:|--------------------:|
| B1_Clinical_Only |     42 |     0.6250 |              0.4940 |     0.4450 |            0.4328 |         0.4940 |              0.8947 |              0.7777 |
| B1_Clinical_Only |    100 |     0.5987 |              0.4403 |     0.3981 |            0.3870 |         0.4403 |              0.8936 |              0.7509 |
| B1_Clinical_Only |   2026 |     0.6579 |              0.4431 |     0.4160 |            0.4017 |         0.4431 |              0.9002 |              0.7767 |
| B2_Derm_Only     |     42 |     0.6974 |              0.5565 |     0.5292 |            0.5164 |         0.5565 |              0.9093 |              0.8511 |
| B2_Derm_Only     |    100 |     0.7105 |              0.6025 |     0.5465 |            0.5192 |         0.6025 |              0.9197 |              0.8913 |
| B2_Derm_Only     |   2026 |     0.6842 |              0.5662 |     0.5001 |            0.4787 |         0.5662 |              0.9178 |              0.8594 |
| B3_Meta_Only     |     42 |     0.5000 |              0.5339 |     0.3952 |            0.4162 |         0.5339 |              0.8642 |              0.7396 |
| B3_Meta_Only     |    100 |     0.4737 |              0.5283 |     0.3601 |            0.3739 |         0.5283 |              0.8677 |              0.7432 |
| B3_Meta_Only     |   2026 |     0.3816 |              0.4737 |     0.2632 |            0.2719 |         0.4737 |              0.8436 |              0.7293 |
| B4_Dual_NoMeta   |     42 |     0.6711 |              0.4578 |     0.4432 |            0.4432 |         0.4578 |              0.9137 |              0.8222 |
| B4_Dual_NoMeta   |    100 |     0.5526 |              0.5658 |     0.4395 |            0.4369 |         0.5658 |              0.8883 |              0.8612 |
| B4_Dual_NoMeta   |   2026 |     0.6645 |              0.5418 |     0.4945 |            0.4765 |         0.5418 |              0.9096 |              0.8417 |
| B5_Dual_Metadata |     42 |     0.6908 |              0.5574 |     0.5145 |            0.4951 |         0.5574 |              0.9159 |              0.8737 |
| B5_Dual_Metadata |    100 |     0.7237 |              0.5550 |     0.5294 |            0.5193 |         0.5550 |              0.9261 |              0.8884 |
| B5_Dual_Metadata |   2026 |     0.7171 |              0.6213 |     0.5564 |            0.5242 |         0.6213 |              0.9239 |              0.8936 |
| B6_PureCBM       |     42 |     0.6250 |              0.5008 |     0.4246 |            0.4089 |         0.5008 |              0.8992 |              0.7924 |
| B6_PureCBM       |    100 |     0.6250 |              0.4904 |     0.4341 |            0.4295 |         0.4904 |              0.9054 |              0.8172 |
| B6_PureCBM       |   2026 |     0.6579 |              0.6049 |     0.5340 |            0.5158 |         0.6049 |              0.9090 |              0.8485 |
| Proposed_Hybrid  |     42 |     0.6974 |              0.5394 |     0.5076 |            0.4901 |         0.5394 |              0.9195 |              0.8565 |
| Proposed_Hybrid  |    100 |     0.7039 |              0.4865 |     0.4746 |            0.4674 |         0.4865 |              0.9151 |              0.8767 |
| Proposed_Hybrid  |   2026 |     0.6513 |              0.4993 |     0.4521 |            0.4403 |         0.4993 |              0.9054 |              0.8334 |

## Class-wise sensitivity/specificity

| Model            | Class                | Sensitivity     | Specificity     |
|:-----------------|:---------------------|:----------------|:----------------|
| B1_Clinical_Only | Basal Cell Carcinoma | 0.4000 ± 0.2000 | 0.9569 ± 0.0142 |
| B1_Clinical_Only | Melanoma             | 0.6762 ± 0.0330 | 0.7892 ± 0.0385 |
| B1_Clinical_Only | Miscellaneous        | 0.5417 ± 0.0722 | 0.9142 ± 0.0297 |
| B1_Clinical_Only | Nevus                | 0.6778 ± 0.0484 | 0.8548 ± 0.0323 |
| B1_Clinical_Only | Seborrheic Keratosis | 0.0000 ± 0.0000 | 0.9658 ± 0.0137 |
| B2_Derm_Only     | Basal Cell Carcinoma | 0.3333 ± 0.1155 | 0.9683 ± 0.0104 |
| B2_Derm_Only     | Melanoma             | 0.6286 ± 0.0495 | 0.8946 ± 0.0049 |
| B2_Derm_Only     | Miscellaneous        | 0.5208 ± 0.0361 | 0.9167 ± 0.0112 |
| B2_Derm_Only     | Nevus                | 0.7815 ± 0.0339 | 0.8602 ± 0.0493 |
| B2_Derm_Only     | Seborrheic Keratosis | 0.6111 ± 0.0962 | 0.9384 ± 0.0119 |
| B3_Meta_Only     | Basal Cell Carcinoma | 1.0000 ± 0.0000 | 0.8594 ± 0.0453 |
| B3_Meta_Only     | Melanoma             | 0.2381 ± 0.0330 | 0.8490 ± 0.0247 |
| B3_Meta_Only     | Miscellaneous        | 0.2292 ± 0.2009 | 0.9632 ± 0.0195 |
| B3_Meta_Only     | Nevus                | 0.5370 ± 0.0819 | 0.8172 ± 0.0406 |
| B3_Meta_Only     | Seborrheic Keratosis | 0.5556 ± 0.0962 | 0.8037 ± 0.0446 |
| B4_Dual_NoMeta   | Basal Cell Carcinoma | 0.4000 ± 0.2000 | 0.9569 ± 0.0283 |
| B4_Dual_NoMeta   | Melanoma             | 0.5619 ± 0.1409 | 0.8803 ± 0.0148 |
| B4_Dual_NoMeta   | Miscellaneous        | 0.6250 ± 0.0625 | 0.9069 ± 0.0042 |
| B4_Dual_NoMeta   | Nevus                | 0.6889 ± 0.0778 | 0.8871 ± 0.0161 |
| B4_Dual_NoMeta   | Seborrheic Keratosis | 0.3333 ± 0.3333 | 0.8881 ± 0.0514 |
| B5_Dual_Metadata | Basal Cell Carcinoma | 0.6000 ± 0.2000 | 0.9705 ± 0.0079 |
| B5_Dual_Metadata | Melanoma             | 0.6857 ± 0.0571 | 0.8803 ± 0.0256 |
| B5_Dual_Metadata | Miscellaneous        | 0.6667 ± 0.0361 | 0.9069 ± 0.0042 |
| B5_Dual_Metadata | Nevus                | 0.7704 ± 0.0064 | 0.8978 ± 0.0246 |
| B5_Dual_Metadata | Seborrheic Keratosis | 0.1667 ± 0.0000 | 0.9543 ± 0.0259 |
| B6_PureCBM       | Basal Cell Carcinoma | 0.5333 ± 0.1155 | 0.9342 ± 0.0336 |
| B6_PureCBM       | Melanoma             | 0.6476 ± 0.0660 | 0.9117 ± 0.0178 |
| B6_PureCBM       | Miscellaneous        | 0.6458 ± 0.0955 | 0.8529 ± 0.0147 |
| B6_PureCBM       | Nevus                | 0.6667 ± 0.0192 | 0.8763 ± 0.0246 |
| B6_PureCBM       | Seborrheic Keratosis | 0.1667 ± 0.1667 | 0.9475 ± 0.0220 |
| Proposed_Hybrid  | Basal Cell Carcinoma | 0.3333 ± 0.1155 | 0.9592 ± 0.0236 |
| Proposed_Hybrid  | Melanoma             | 0.6571 ± 0.0756 | 0.8775 ± 0.0261 |
| Proposed_Hybrid  | Miscellaneous        | 0.5625 ± 0.0625 | 0.9093 ± 0.0297 |
| Proposed_Hybrid  | Nevus                | 0.7667 ± 0.0294 | 0.8710 ± 0.0279 |
| Proposed_Hybrid  | Seborrheic Keratosis | 0.2222 ± 0.0962 | 0.9498 ± 0.0040 |