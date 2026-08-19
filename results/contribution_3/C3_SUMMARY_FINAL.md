# Contribution 3 — Rigorous Experimental Evaluation

## 1. Objective

Contribution 3 establishes a controlled and reproducible evaluation protocol for the Derm7pt multimodal classification study.

The contribution focuses on:

- model and hyperparameter selection using Validation only;
- evaluation across three locked random seeds;
- independent Test-set evaluation;
- macro-level classification metrics;
- per-class sensitivity and specificity;
- uncertainty reporting across training runs;
- stratified bootstrap confidence intervals.

OOD detection and conformal prediction are outside the scope of this contribution.

---

## 2. Data-Split Protocol

The dataset is divided into four disjoint subsets:

- Train: 605 samples
- Validation: 152 samples
- Calibration: 102 samples
- Test: 152 samples

The split procedure is group-aware using `case_num`.

Therefore, the split is described as a:

> case-level grouped split

It should not be called a patient-level split unless `case_num` is independently verified to uniquely identify patients.

The Test set is not used for:

- training;
- early stopping;
- checkpoint selection;
- alpha selection.

The Calibration split is reserved and is not currently used for calibration or model selection.

---

## 3. Model Selection Protocol

Checkpoint selection is performed on the Validation set.

The main checkpoint-selection criterion is:

> Validation disease Macro-F1

The concept-loss weight alpha is also selected using Validation performance only.

The selected value used in the final Concept Bottleneck experiments is:

> alpha = 2.0

The Test set is evaluated only after model and hyperparameter selection has been completed.

---

## 4. Repeated Training

All main experiments are evaluated using three locked seeds:

- 42
- 100
- 2026

The primary uncertainty reporting across independent training runs is:

> Mean ± sample standard deviation

This is the main variability estimate reported for model comparisons.

---

## 5. Models Evaluated

Seven main model configurations are evaluated:

1. B1_Clinical_Only
2. B2_Derm_Only
3. B3_Meta_Only
4. B4_Dual_NoMeta
5. B5_Dual_Metadata
6. B6_PureCBM
7. Proposed_Hybrid

---

## 6. Independent Test Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Macro Precision | Macro Recall | Macro Specificity | OVR AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 Clinical Only | 0.6272 ± 0.0297 | 0.4591 ± 0.0302 | 0.4197 ± 0.0237 | 0.4072 ± 0.0234 | 0.4591 ± 0.0302 | 0.8962 ± 0.0035 | 0.7684 ± 0.0152 |
| B2 Derm Only | 0.6974 ± 0.0132 | 0.5751 ± 0.0243 | 0.5253 ± 0.0234 | 0.5048 ± 0.0226 | 0.5751 ± 0.0243 | 0.9156 ± 0.0055 | 0.8672 ± 0.0212 |
| B3 Meta Only | 0.4518 ± 0.0622 | 0.5120 ± 0.0333 | 0.3395 ± 0.0684 | 0.3540 ± 0.0742 | 0.5120 ± 0.0333 | 0.8585 ± 0.0130 | 0.7374 ± 0.0072 |
| B4 Dual No Metadata | 0.6294 ± 0.0666 | 0.5218 ± 0.0567 | 0.4591 ± 0.0307 | 0.4522 ± 0.0213 | 0.5218 ± 0.0567 | 0.9039 ± 0.0136 | 0.8417 ± 0.0195 |
| **B5 Dual Metadata** | **0.7105 ± 0.0174** | **0.5779 ± 0.0376** | **0.5334 ± 0.0213** | **0.5129 ± 0.0156** | **0.5779 ± 0.0376** | **0.9220 ± 0.0054** | **0.8852 ± 0.0103** |
| B6 Pure CBM | 0.5680 ± 0.1405 | 0.4665 ± 0.0706 | 0.4018 ± 0.0847 | 0.4180 ± 0.0567 | 0.4665 ± 0.0706 | 0.8942 ± 0.0272 | 0.7495 ± 0.0197 |
| Proposed Hybrid | 0.6623 ± 0.0274 | 0.4779 ± 0.0230 | 0.4489 ± 0.0287 | 0.4434 ± 0.0179 | 0.4779 ± 0.0230 | 0.9070 ± 0.0073 | 0.8465 ± 0.0238 |

---

## 7. Main Disease-Classification Finding

Among the evaluated configurations, **B5_Dual_Metadata** provides the strongest numerical disease-classification performance.

It achieves:

- Accuracy: 0.7105 ± 0.0174
- Balanced Accuracy: 0.5779 ± 0.0376
- Macro F1: 0.5334 ± 0.0213
- OVR AUROC: 0.8852 ± 0.0103

Therefore, the Proposed Hybrid model should **not** be described as the best disease classifier.

Instead, its role is evaluated as a trade-off between disease prediction and concept-based interpretability.

No statistical significance claim is made between models based solely on the reported mean ± SD values.

---

## 8. Modality Findings

The evaluation provides several numerical observations.

### Clinical versus Dermoscopy

B2_Derm_Only substantially outperforms B1_Clinical_Only in Macro F1:

- Clinical only: 0.4197 ± 0.0237
- Dermoscopy only: 0.5253 ± 0.0234

This indicates that dermoscopic images contain stronger disease-discriminative information in this experimental setting.

### Metadata Alone

Metadata alone provides substantially weaker Macro F1:

- B3_Meta_Only: 0.3395 ± 0.0684

Therefore, metadata is not sufficient as a standalone modality for disease classification.

### Metadata with Dual Images

Adding metadata to the dual-image model improves the numerical result:

- B4_Dual_NoMeta Macro F1: 0.4591 ± 0.0307
- B5_Dual_Metadata Macro F1: 0.5334 ± 0.0213

This supports the usefulness of structured metadata when combined with image modalities.

These are descriptive comparisons and are not presented as formal significance tests.

---

## 9. Concept Bottleneck Results

The corrected Concept Bottleneck experiments produce:

- B6_PureCBM Macro F1: 0.4018 ± 0.0847
- Proposed_Hybrid Macro F1: 0.4489 ± 0.0287

The Hybrid model is numerically stronger and more stable than the Pure CBM for disease prediction.

However, both remain below the black-box B5_Dual_Metadata model in disease Macro F1.

This demonstrates the predictive-performance versus interpretability trade-off explored further in Contribution 2.

---

## 10. Per-Class Evaluation

Sensitivity and specificity are evaluated separately for all five disease classes and all seven model configurations.

The complete results are stored in:

`c3_per_class_test_metrics.csv`

The per-class results reveal substantial variability across disease classes.

This is especially important because the Test set is class-imbalanced and some classes contain very few examples.

Therefore, overall accuracy alone is not sufficient for evaluating model performance.

Macro metrics and per-class metrics are retained as the principal reporting strategy.

---

## 11. Bootstrap Confidence Intervals

For the Proposed Hybrid model, stratified percentile bootstrap confidence intervals are computed using:

- 1000 bootstrap replicates;
- fixed bootstrap random state;
- within-class resampling;
- 95% percentile confidence intervals.

Macro-F1 results are:

| Analysis | Macro F1 | 95% CI |
|---|---:|---:|
| Seed 42 | 0.4707 | [0.3681, 0.5857] |
| Seed 100 | 0.4596 | [0.3761, 0.5652] |
| Seed 2026 | 0.4164 | [0.3347, 0.5080] |
| 3-seed probability ensemble | 0.4383 | [0.3577, 0.5279] |

The bootstrap intervals quantify Test-sample uncertainty.

They do not replace the primary across-training-run variability estimate.

---

## 12. Probability Ensemble

The probability ensemble averages the class probabilities from the three independently trained Proposed Hybrid models.

Its Test results are:

- Accuracy: 0.6645
- Balanced Accuracy: 0.4627
- Macro F1: 0.4383

The probability ensemble is treated only as a:

> secondary analysis

The primary reporting remains:

> Mean ± sample SD over the three training seeds

---

## 13. Evaluation Strengths

Contribution 3 establishes the following safeguards:

- fixed Train / Validation / Calibration / Test partitions;
- grouped split at the case level;
- no Test-based checkpoint selection;
- no Test-based alpha selection;
- three locked training seeds;
- macro-level classification metrics;
- independent Test evaluation;
- per-class sensitivity and specificity;
- bootstrap confidence intervals;
- fixed bootstrap random state;
- explicit separation between training-run variability and Test-sample uncertainty.

---

## 14. Limitations

The following limitations must be retained:

1. The independent Test set contains only 152 samples.
2. Disease classes are strongly imbalanced.
3. Some classes contain very few Test examples.
4. Mean ± SD is based on only three training seeds.
5. Bootstrap intervals quantify Test-sample variability, not uncertainty over all possible training runs.
6. No formal statistical significance test between model configurations is reported.
7. The Calibration set is reserved but has not been used for probability calibration.
8. OOD detection has not been evaluated.
9. Conformal prediction has not been implemented.
10. The grouped split should not be described as patient-level without independent verification of `case_num`.

---

## 15. Conclusion

Contribution 3 provides a rigorous experimental evaluation framework for comparing multimodal and Concept Bottleneck models on Derm7pt.

The strongest numerical disease-classification result is obtained by B5_Dual_Metadata with a Macro F1 of:

> 0.5334 ± 0.0213

Dermoscopy provides stronger disease-discriminative performance than clinical images alone, while structured metadata improves the dual-image model when used jointly with image features.

The Proposed Hybrid model does not achieve the highest disease-classification score, but provides a concept-based architecture whose interpretability properties are analyzed separately in Contribution 2.

Overall, the evaluation protocol emphasizes independent Test assessment, repeated training, macro metrics, per-class analysis, and explicit uncertainty reporting while avoiding unsupported claims about statistical significance, calibration, OOD detection, or patient-level splitting.

**Contribution 3: Rigorous Experimental Evaluation — COMPLETED.**
