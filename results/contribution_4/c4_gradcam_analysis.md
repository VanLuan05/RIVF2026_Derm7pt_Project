# Contribution 4 — Grad-CAM Qualitative Analysis

## 1. Protocol

Grad-CAM was generated using the pre-specified Proposed Hybrid model with seed 42.

Examples were processed in deterministic Test-set order (`shuffle=False`).
The procedure retained at most one correctly classified and one incorrectly
classified example for each true disease class.

The examples were therefore not selected according to Test confidence or
performance score.

A total of 10 examples were obtained:

- 5 correctly classified examples;
- 5 incorrectly classified examples;
- all 5 disease classes represented in both groups.

Grad-CAM was generated separately for:

- the clinical-image branch;
- the dermoscopic-image branch.

## 2. Main Qualitative Observations

The visual explanations show that attention often overlaps with the lesion
region, particularly in the dermoscopic modality.

However, several examples also exhibit activation outside the lesion,
including image borders, surrounding skin, and ruler markings.

This behavior is visible in both correctly and incorrectly classified cases.

Therefore, Grad-CAM should not be interpreted as proof that the network always
uses clinically meaningful evidence.

## 3. Success Cases

Several correctly classified samples exhibit Grad-CAM activation that overlaps
substantially with the visible lesion.

The Miscellaneous success example shows relatively lesion-centered activation
in both clinical and dermoscopic views.

The Seborrheic Keratosis and Basal Cell Carcinoma success examples also show
substantial attention near lesion regions.

However, some correctly classified examples still contain off-lesion
activation. In particular, the Nevus success example shows noticeable clinical
activation near ruler markings, while its dermoscopic attention is more
lesion-centered.

Thus, correct classification alone does not guarantee anatomically or
clinically plausible attention.

## 4. Failure Cases

Failure examples reveal two different behaviors.

First, some incorrect predictions still show substantial attention to the
visible lesion. For example, the Nevus-to-Melanoma error contains
lesion-centered activation in both modalities.

This indicates that plausible lesion localization is not sufficient for
correct disease discrimination.

Second, some failures show attention toward possible non-lesion artifacts.
The Miscellaneous-to-Seborrheic-Keratosis example exhibits strong clinical
activation near ruler/image-border regions.

The Melanoma-to-Nevus example also shows a clinical hotspot toward the image
edge, while the dermoscopic branch is more focused on the lesion.

These observations suggest that the model may occasionally exploit
non-lesion visual cues.

## 5. High-Confidence Errors

Some misclassified examples have high predicted confidence.

Examples include:

- Basal Cell Carcinoma predicted as Miscellaneous: approximately 87.2%;
- Seborrheic Keratosis predicted as Basal Cell Carcinoma: approximately 86.0%;
- Nevus predicted as Melanoma: approximately 76.9%.

These cases illustrate that high softmax confidence does not necessarily imply
correct classification or clinically plausible visual evidence.

This observation is consistent with the uncertainty analysis in Contribution 5.

## 6. Modality-Specific Behavior

The dermoscopic branch frequently provides more lesion-centered activation than
the clinical branch in the inspected examples.

The clinical branch more often exhibits activation over surrounding skin,
rulers, or image borders.

This is a qualitative observation from the selected deterministic examples and
is not presented as a quantitative comparison between modalities.

## 7. Interpretation

The Grad-CAM analysis provides qualitative support for three conclusions:

1. The Proposed Hybrid model often attends to lesion-related visual regions.
2. Attention patterns differ between clinical and dermoscopic modalities.
3. Non-lesion and artifact-related attention remains present in several cases.

Grad-CAM therefore serves primarily as a diagnostic visualization tool for
inspecting model behavior rather than as evidence of causal or clinically
validated reasoning.

## 8. Limitations

- Only 10 deterministic examples are visualized.
- Grad-CAM is qualitative.
- No lesion segmentation masks are available for quantitative localization.
- No expert dermatologist assessment of Grad-CAM regions was performed.
- Attention localization does not establish causal feature use.
- A correct prediction can still contain implausible attention.
- An incorrect prediction can still attend to the visible lesion.

Therefore, no quantitative localization accuracy or clinical-faithfulness claim
is made.
