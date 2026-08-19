# CONTRIBUTION 4 - FINAL SUMMARY

## 1. Objective

Contribution 4 investigates the visual explainability of the Proposed Hybrid
model using Grad-CAM.

The objective is to qualitatively inspect which image regions contribute to
the model's disease prediction in:

- clinical images;
- dermoscopic images.

This contribution is intended as a diagnostic interpretability analysis.

It does **not** claim:

- causal explanation;
- quantitative lesion localization;
- dermatologist-validated explanation quality;
- segmentation accuracy;
- clinically validated reasoning.

---

## 2. Model and Protocol

Grad-CAM is generated from:

> Proposed_Hybrid, seed = 42

The seed is pre-specified through:

> `Config.GRADCAM_SEED = 42`

It is not selected according to Test performance.

The independent Test set is processed with:

- `batch_size = 1`;
- `shuffle = False`;
- deterministic sample order.

The selection procedure retains at most:

- one correctly classified example per true disease class;
- one incorrectly classified example per true disease class.

Examples are therefore not selected according to confidence or Test score.

---

## 3. Disease-Class Coverage

A total of 10 examples are visualized:

- 5 Success cases;
- 5 Failure cases.

All five disease classes are represented in both groups:

- Basal Cell Carcinoma;
- Melanoma;
- Miscellaneous;
- Nevus;
- Seborrheic Keratosis.

Thus:

> Success classes represented = 5/5

and:

> Failure classes represented = 5/5

---

## 4. Grad-CAM Generation

Grad-CAM is generated separately for the two image branches.

### Clinical branch

The clinical image is treated as the variable input while:

- dermoscopic image;
- metadata

remain fixed.

### Dermoscopic branch

The dermoscopic image is treated as the variable input while:

- clinical image;
- metadata

remain fixed.

Grad-CAM is extracted from the final convolutional region of the corresponding
ResNet-50 branch.

The target class is the model's predicted disease class.

---

## 5. Main Qualitative Observations

The inspected examples show that Grad-CAM activation often overlaps with the
visible lesion region.

This behavior is particularly noticeable in several dermoscopic examples.

However, activation outside the lesion is also present.

Observed non-lesion regions include:

- surrounding skin;
- image borders;
- ruler markings;
- peripheral image regions.

Therefore, Grad-CAM does not demonstrate that the model exclusively relies on
clinically meaningful lesion features.

---

## 6. Success Cases

Several correctly classified cases show substantial lesion-centered activation.

Examples include:

- Basal Cell Carcinoma classified correctly;
- Miscellaneous classified correctly;
- Seborrheic Keratosis classified correctly.

The Miscellaneous success example shows relatively lesion-centered attention
in both clinical and dermoscopic views.

The dermoscopic branch of the Nevus success example also focuses substantially
on the lesion.

However, the corresponding clinical Grad-CAM contains noticeable activation
near ruler markings.

This demonstrates that:

> correct classification does not necessarily imply fully plausible visual
> attention.

---

## 7. Failure Cases

Incorrect predictions exhibit multiple types of behavior.

### Lesion-centered but incorrect prediction

The Nevus-to-Melanoma failure shows substantial activation around the visible
lesion in both modalities.

Therefore:

> plausible lesion localization is not sufficient for correct disease
> discrimination.

### Possible artifact-related attention

The Miscellaneous-to-Seborrheic-Keratosis failure shows strong clinical
activation close to ruler and image-border regions.

The Melanoma-to-Nevus example also contains a strong clinical hotspot toward
the image periphery, while its dermoscopic Grad-CAM is more lesion-centered.

These examples indicate that the model may occasionally attend to non-lesion
visual cues.

---

## 8. High-Confidence Failure Cases

Several incorrect predictions have relatively high softmax confidence.

Examples include:

| True Class | Predicted Class | Confidence |
|---|---|---:|
| Basal Cell Carcinoma | Miscellaneous | 0.8722 |
| Seborrheic Keratosis | Basal Cell Carcinoma | 0.8598 |
| Nevus | Melanoma | 0.7687 |
| Miscellaneous | Seborrheic Keratosis | 0.7660 |
| Melanoma | Nevus | 0.4675 |

These examples demonstrate that high predictive confidence does not guarantee:

- correct classification;
- lesion-centered attention;
- clinically plausible visual evidence.

This observation is consistent with the uncertainty analysis reported in
Contribution 5.

---

## 9. Clinical versus Dermoscopic Attention

Across the inspected examples, the dermoscopic branch frequently exhibits
more lesion-centered activation than the clinical branch.

The clinical branch more often displays activation over:

- surrounding skin;
- rulers;
- borders;
- peripheral image regions.

This should be treated as a qualitative observation only.

No quantitative statistical comparison between clinical and dermoscopic
Grad-CAM localization is performed.

---

## 10. Interpretation

The Grad-CAM analysis supports three qualitative conclusions.

### 1. Lesion-related attention is often present

The Proposed Hybrid model frequently activates around visible lesion regions.

### 2. Modalities exhibit different attention patterns

Clinical and dermoscopic branches do not necessarily focus on identical image
regions.

### 3. Spurious or non-lesion attention remains possible

Some examples contain substantial activation over rulers, borders, surrounding
skin, or peripheral regions.

Grad-CAM should therefore be interpreted as:

> a post-hoc diagnostic visualization of model behavior

rather than proof of causal or clinically validated reasoning.

---

## 11. Reproducibility Evidence

A manifest is stored for every Grad-CAM example.

For each sample it records:

- Test sample index;
- `case_num`;
- `case_id`;
- Success / Failure status;
- true disease;
- predicted disease;
- prediction confidence;
- clinical image path;
- dermoscopic image path;
- Grad-CAM output filename;
- Grad-CAM seed.

This allows every displayed example to be traced back to the fixed Test split.

---

## 12. Limitations

The following limitations must be retained when reporting Contribution 4:

1. Only 10 deterministic examples are visualized.
2. Grad-CAM is a qualitative post-hoc method.
3. No lesion segmentation masks are available for quantitative localization.
4. No dermatologist assessment of Grad-CAM maps is performed.
5. Attention localization does not establish causal feature usage.
6. Correct predictions may still contain non-lesion attention.
7. Incorrect predictions may still attend to the visible lesion.
8. No quantitative comparison between clinical and dermoscopic localization is performed.
9. Grad-CAM results are shown for one pre-specified seed.
10. The analysis must not be interpreted as clinical validation of model reasoning.

---

## 13. Conclusion

Contribution 4 provides a qualitative visual explainability analysis of the
Proposed Hybrid model using Grad-CAM.

The inspected cases show that the model often attends to lesion-related
regions, particularly in dermoscopic images.

At the same time, Grad-CAM reveals several examples of attention toward
non-lesion regions such as rulers, borders, and surrounding skin.

The analysis also demonstrates that:

- plausible lesion localization does not guarantee correct classification;
- correct classification does not guarantee fully plausible attention;
- high confidence does not guarantee correct or clinically meaningful reasoning.

Therefore, Grad-CAM is used as a diagnostic interpretability tool rather than
as proof of clinical faithfulness.

**Contribution 4: Visual Explainability with Grad-CAM — COMPLETED.**

---

## 14. Files

### Source

- `src/gradcam_vis.py`

### Results

- `results/contribution_4/C4_SUMMARY_FINAL.md`
- `results/contribution_4/c4_gradcam_analysis.md`
- `results/contribution_4/c4_gradcam_manifest.csv`
- `results/contribution_4/c4_gradcam_contact_sheet.png`

### Grad-CAM Images

- `outputs/gradcam_results/`

The folder contains:

- 5 Success images;
- 5 Failure images.
