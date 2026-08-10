# Concept Prediction Performance on Test Set

Threshold-dependent metrics use a fixed 0.5 threshold. AUROC is threshold-free. Values are mean ± sample SD across available seeds.

| Model           | Concept               |   Test Prevalence | AUROC           | F1              | Precision       | Recall          |
|:----------------|:----------------------|------------------:|:----------------|:----------------|:----------------|:----------------|
| B6_PureCBM      | Pigment Network       |            0.2434 | 0.7939 ± 0.0427 | 0.5427 ± 0.0427 | 0.5672 ± 0.0305 | 0.5315 ± 0.1092 |
| B6_PureCBM      | Streaks               |            0.2039 | 0.8509 ± 0.0401 | 0.5849 ± 0.0354 | 0.5350 ± 0.0881 | 0.6667 ± 0.0986 |
| B6_PureCBM      | Pigmentation          |            0.3092 | 0.7753 ± 0.0312 | 0.5500 ± 0.0304 | 0.5292 ± 0.0256 | 0.5816 ± 0.0959 |
| B6_PureCBM      | Regression Structures |            0      | N/A             | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| B6_PureCBM      | Dots & Globules       |            0.4013 | 0.7841 ± 0.0312 | 0.6814 ± 0.0361 | 0.6754 ± 0.0496 | 0.6885 ± 0.0328 |
| B6_PureCBM      | Blue-whitish Veil     |            0.1645 | 0.8498 ± 0.0250 | 0.5440 ± 0.0612 | 0.4547 ± 0.0442 | 0.6800 ± 0.1058 |
| B6_PureCBM      | Vascular Structures   |            0.0263 | 0.7382 ± 0.0310 | 0.0952 ± 0.1650 | 0.1111 ± 0.1925 | 0.0833 ± 0.1443 |
| Proposed_Hybrid | Pigment Network       |            0.2434 | 0.8272 ± 0.0087 | 0.5625 ± 0.0271 | 0.6681 ± 0.0428 | 0.4865 ± 0.0270 |
| Proposed_Hybrid | Streaks               |            0.2039 | 0.8848 ± 0.0303 | 0.6020 ± 0.0208 | 0.6062 ± 0.0475 | 0.6022 ± 0.0493 |
| Proposed_Hybrid | Pigmentation          |            0.3092 | 0.8186 ± 0.0231 | 0.6243 ± 0.0336 | 0.6198 ± 0.0408 | 0.6312 ± 0.0535 |
| Proposed_Hybrid | Regression Structures |            0      | N/A             | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Proposed_Hybrid | Dots & Globules       |            0.4013 | 0.8296 ± 0.0168 | 0.6929 ± 0.0281 | 0.7003 ± 0.0597 | 0.6885 ± 0.0328 |
| Proposed_Hybrid | Blue-whitish Veil     |            0.1645 | 0.8614 ± 0.0173 | 0.5105 ± 0.0662 | 0.5144 ± 0.0718 | 0.5067 ± 0.0611 |
| Proposed_Hybrid | Vascular Structures   |            0.0263 | 0.7173 ± 0.0319 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |