# Contribution 2 — Concept Bottleneck Modeling and Concept Intervention

## 1. Objective

Contribution 2 investigates Concept Bottleneck Models (CBMs) using seven clinically meaningful Derm7pt concepts as intermediate representations for skin-lesion diagnosis.

The evaluated CBM variants are:

- **B6_PureCBM**: disease prediction depends on the predicted concept bottleneck.
- **Proposed_Hybrid**: disease prediction uses both latent multimodal features and predicted concepts.

All reported results use the corrected Derm7pt concept encoding, the fixed Train/Validation/Calibration/Test splits, and the three locked seeds: 42, 100, and 2026.

The concept-loss weight was selected on the Validation set only, with:

- **alpha = 2.0**

The independent Test set contains 152 samples.

---

## 2. Concept Prediction Performance

Concept prediction was evaluated using the seven binary Derm7pt concepts with a fixed sigmoid threshold of 0.5.

| Model | Macro Concept F1 | Macro Concept AUROC |
|---|---:|---:|
| B6_PureCBM | 0.5171 ± 0.0633 | 0.7868 ± 0.0597 |
| Proposed_Hybrid | **0.5606 ± 0.0247** | **0.8437 ± 0.0136** |

The Proposed_Hybrid model achieved numerically higher concept-level Macro-F1 and Macro-AUROC and showed lower variability across the three seeds.

These differences are reported descriptively and should not be interpreted as statistically significant without a dedicated statistical test.

---

## 3. Disease Classification Performance

Disease classification was evaluated on the independent Test set.

| Model | Accuracy | Balanced Accuracy | Macro F1 | One-vs-Rest AUROC |
|---|---:|---:|---:|---:|
| B6_PureCBM | 0.5680 ± 0.1405 | 0.4665 ± 0.0706 | 0.4018 ± 0.0847 | 0.7495 ± 0.0197 |
| Proposed_Hybrid | **0.6623 ± 0.0274** | **0.4779 ± 0.0230** | **0.4489 ± 0.0287** | **0.8465 ± 0.0238** |

The Proposed_Hybrid model produced numerically stronger and more stable disease classification performance, particularly in Macro-F1 and AUROC.

---

## 4. End-to-End Soft Concept Intervention

A diagnostic oracle intervention replaced predicted concept probabilities with soft ground-truth concept values of 0.05/0.95 while keeping the same trained model.

| Model | Macro F1 AI | Macro F1 Oracle | Delta |
|---|---:|---:|---:|
| B6_PureCBM | 0.4018 ± 0.0847 | 0.2281 ± 0.0743 | -0.1737 |
| Proposed_Hybrid | 0.4489 ± 0.0287 | 0.4472 ± 0.0268 | -0.0017 |

Balanced Accuracy also decreased:

- B6_PureCBM: 0.4665 ± 0.0706 -> 0.2476 ± 0.0129
- Proposed_Hybrid: 0.4779 ± 0.0230 -> 0.4771 ± 0.0223

Therefore, full soft-oracle concept replacement did **not** improve disease classification.

This experiment is an oracle diagnostic rather than a prospective clinician intervention study.

---

## 5. Sequential Concept-Only Diagnostic

A downstream logistic regression classifier was trained using only the seven concept outputs, without a metadata bypass.

### Representation comparison

| Representation | Accuracy | Macro F1 |
|---|---:|---:|
| Soft predicted concepts | **0.6711 ± 0.0570** | **0.4451 ± 0.0054** |
| Hard predicted concepts | 0.6316 ± 0.0914 | 0.3999 ± 0.0508 |
| Ground-truth concepts | 0.3947 ± 0.0000 | 0.3725 ± 0.0000 |

Soft predicted concept probabilities consistently outperformed the corresponding thresholded binary concepts.

This provides evidence that the continuous bottleneck probabilities contain disease-relevant information beyond the intended binary clinical concept states.

This finding should be described as evidence of a **soft-bottleneck shortcut or concept-faithfulness limitation**, rather than conventional dataset leakage.

---

## 6. Hard-Bottleneck Intervention

To remove the soft-probability distribution effect, a hard-bottleneck Sequential CBM diagnostic was evaluated.

The disease classifier was trained on binary predicted concepts only. At Test time, the same disease classifier was evaluated first with predicted binary concepts and then with all seven ground-truth binary concepts.

| Metric | Predicted Concepts | GT Intervention | Mean Delta |
|---|---:|---:|---:|
| Macro F1 | 0.3999 ± 0.0508 | 0.2816 ± 0.0073 | -0.1183 |
| Balanced Accuracy | 0.4240 ± 0.0347 | 0.2819 ± 0.0154 | -0.1420 |

Macro-F1 decreased in all three seeds:

- Seed 42: -0.0738
- Seed 100: -0.1186
- Seed 2026: -0.1626

Thus, the negative intervention effect cannot be explained solely by replacing soft probabilities with values from a different numerical distribution.

---

## 7. Interpretation

The experiments support three main observations.

First, the Proposed_Hybrid architecture provides better numerical concept prediction and disease classification performance than the PureCBM variant.

Second, predicted concept probabilities are informative for disease classification, but their predictive information is not fully aligned with the intended binary Derm7pt semantics.

Third, replacing predicted concepts with ground-truth concepts does not improve disease prediction. This indicates limited concept faithfulness and limited intervenability of the current jointly trained bottleneck.

The results therefore suggest a trade-off between predictive utility and semantic faithfulness in the learned concept representation.

---

## 8. Limitations

Several limitations should be considered:

1. The Test set is relatively small (n = 152).
2. Results are summarized over only three fixed seeds.
3. Mean ± sample SD does not establish statistical significance.
4. The soft-oracle intervention is a diagnostic analysis and not a prospective clinician intervention.
5. The PureCBM is jointly optimized with both disease and concept losses, allowing disease supervision to influence the continuous concept representation.
6. Ground-truth binary Derm7pt concepts may not contain all information needed for five-class disease discrimination.
7. The negative intervention results should therefore be interpreted as evidence of limited concept faithfulness rather than evidence that clinical concepts themselves are uninformative.

---

## 9. Contribution 2 Summary

Contribution 2 establishes a clinically structured Concept Bottleneck framework based on seven Derm7pt concepts and evaluates both concept prediction quality and concept intervenability.

The Proposed_Hybrid model achieved the strongest overall concept and disease performance, while intervention analyses revealed that the learned concept bottleneck is not fully semantically faithful.

These findings provide both quantitative interpretability and an explicit analysis of the limitations of concept-based intervention in the proposed multimodal diagnostic framework.
