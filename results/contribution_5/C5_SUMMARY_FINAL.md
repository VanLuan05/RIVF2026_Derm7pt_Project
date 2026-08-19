# Contribution 5 — Prediction Confidence and Uncertainty Analysis

## 1. Objective

Contribution 5 evaluates whether predictive uncertainty from the Proposed Hybrid model can identify difficult or potentially incorrect predictions.

The analysis is retrospective and is performed only on the fixed Test set.

This contribution does **not** claim to implement:

- out-of-distribution (OOD) detection;
- conformal prediction;
- prospective clinical triage;
- a complete active-learning training loop.

The three-seed probability ensemble is treated as a secondary uncertainty analysis.

---

## 2. Experimental Setup

Model:

- Proposed Hybrid Concept Bottleneck Model
- Seeds: 42, 100, 2026
- Test samples: 152
- Five disease classes

Per-sample uncertainty measures:

- maximum softmax confidence;
- predictive entropy;
- normalized entropy;
- probability margin;
- seed disagreement rate;
- probability standard deviation across seeds;
- mutual information.

The three seed probabilities are averaged to obtain the ensemble prediction.

---

## 3. Prediction Confidence

| Source | Accuracy | Mean Confidence | Mean Normalized Entropy | Mean Margin |
|---|---:|---:|---:|---:|
| Seed 42 | 0.6711 | 0.7462 | 0.4604 | 0.6081 |
| Seed 100 | 0.6842 | 0.7347 | 0.4645 | 0.5879 |
| Seed 2026 | 0.6316 | 0.7120 | 0.5129 | 0.5656 |
| 3-seed probability ensemble | 0.6645 | 0.7010 | 0.5259 | 0.5444 |

The ensemble is used as a secondary uncertainty analysis rather than as the primary estimate of training variability.

---

## 4. Correct versus Incorrect Predictions

The ensemble produced:

- Correct predictions: 101
- Incorrect predictions: 51

| Metric | Correct | Incorrect |
|---|---:|---:|
| Confidence | 0.7716 | 0.5612 |
| Normalized entropy | 0.4495 | 0.6771 |
| Margin | 0.6528 | 0.3297 |
| Seed disagreement rate | 0.0561 | 0.2092 |
| Mean probability SD | 0.0530 | 0.0956 |
| Maximum probability SD | 0.1198 | 0.2052 |
| Mutual information | 0.0556 | 0.1134 |

Incorrect predictions therefore show, on average:

- lower confidence;
- higher predictive entropy;
- smaller class-probability margins;
- greater disagreement between independently trained seeds;
- larger probability variability across seeds.

These observations indicate that model uncertainty is associated with prediction errors on the Test set.

---

## 5. Error Detection

Prediction error is treated as the positive class.

| Uncertainty score | Error-detection AUROC |
|---|---:|
| 1 - Margin | **0.8160** |
| 1 - Confidence | 0.8115 |
| Normalized entropy | 0.7929 |
| Mean probability SD | 0.7626 |
| Mutual information | 0.7523 |
| Maximum probability SD | 0.7428 |
| Seed disagreement rate | 0.6906 |

The probability margin provides the strongest numerical error-discrimination performance among the evaluated uncertainty measures.

However, these AUROC values are descriptive Test-set results and no statistical significance comparison between uncertainty measures is claimed.

---

## 6. Selective Prediction

Selective prediction was evaluated by rejecting the most uncertain samples and measuring performance on the retained subset.

Using **1 - margin** as the uncertainty score:

| Coverage | Retained Samples | Accuracy | Macro-F1 | Error Rate |
|---|---:|---:|---:|---:|
| 100% | 152 | 0.6645 | 0.4383 | 0.3355 |
| 90% | 137 | 0.7299 | 0.4997 | 0.2701 |
| 80% | 122 | 0.7541 | 0.5283 | 0.2459 |
| 70% | 107 | 0.7944 | **0.5778** | 0.2056 |
| 60% | 92 | 0.8587 | 0.5646 | 0.1413 |
| 50% | 76 | **0.8684** | 0.5068 | **0.1316** |

Accuracy increases and error rate decreases as the most uncertain cases are removed.

This supports the usefulness of uncertainty for identifying difficult predictions.

Macro-F1, however, does not improve monotonically.

---

## 7. Class-Coverage Limitation

Selective rejection changes the disease-class distribution of the retained subset.

For example, at 50% coverage:

- Basal Cell Carcinoma decreases from 5 to 1 sample;
- Melanoma decreases from 35 to 14;
- Miscellaneous decreases from 16 to 7;
- Nevus decreases from 90 to 53;
- Seborrheic Keratosis decreases from 6 to 1.

The strong reduction of minority-class samples explains why Macro-F1 can decrease even while overall accuracy continues to improve.

Therefore, selective prediction results should not be interpreted solely using accuracy.

---

## 8. Uncertainty Ranking

All 152 Test samples are ranked according to uncertainty.

The ranking prioritizes:

1. high normalized entropy;
2. high seed disagreement;
3. low probability margin.

Most of the highest-ranked uncertain samples are prediction errors, although some correctly classified samples also exhibit high uncertainty.

This ranking may be useful for retrospective inspection or for motivating future active-learning experiments.

It is **not** presented as evidence of an implemented active-learning training procedure.

---

## 9. Interpretation

The experiments provide evidence that predictive uncertainty contains useful information about model reliability.

In particular:

- incorrect predictions have systematically less confident probability distributions;
- probability margin achieves an error-detection AUROC of 0.8160;
- uncertainty-based selective prediction reduces the observed error rate;
- disagreement between independently trained seeds is greater for incorrect predictions.

These findings suggest that uncertainty estimates may help identify samples that warrant additional review.

However, uncertainty-based rejection also changes class coverage and can disproportionately remove minority-class examples.

---

## 10. Limitations

The following limitations must be retained when reporting Contribution 5:

1. Test set size is only 152 samples.
2. Several disease classes contain very few Test examples.
3. The analysis is retrospective.
4. No uncertainty threshold was prospectively validated.
5. No external or OOD dataset was evaluated.
6. OOD detection is therefore not claimed.
7. Conformal prediction was not implemented.
8. Active learning was not performed as an iterative retraining experiment.
9. The three-seed probability ensemble is a secondary analysis.
10. Selective prediction changes the class distribution of retained samples.

---

## 11. Conclusion

Contribution 5 shows that uncertainty measures from the Proposed Hybrid model are informative for retrospective prediction-error analysis.

Among the evaluated measures, probability margin provides the strongest numerical error-detection result.

Selective prediction demonstrates a clear accuracy-risk trade-off: rejecting uncertain cases substantially lowers the observed error rate on retained samples.

At the same time, class-coverage analysis shows that this improvement must be interpreted cautiously because minority classes are disproportionately reduced at lower coverage.

The results therefore support uncertainty estimation as a useful model-reliability diagnostic while avoiding claims of completed OOD detection, conformal prediction, or prospective active learning.
