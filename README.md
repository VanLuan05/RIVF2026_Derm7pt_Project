RIVF2026 Derm7pt — Multimodal Hybrid Concept Bottleneck Model

1. Research scope

This project studies five-class skin-lesion classification on Derm7pt using:

clinical images;

dermoscopy images;

structured metadata;

seven clinically motivated Derm7pt concepts;

a Hybrid Concept Bottleneck Model (Hybrid CBM);

Grad-CAM and oracle concept analyses for interpretability.

The paper-ready scope is intentionally limited to three main contributions:

Multimodal learning with metadata — quantify the value of clinical, dermoscopic and metadata streams.

Concept-based modeling — compare black-box multimodal baselines, Pure CBM and Proposed Hybrid CBM.

Rigorous ablation/evaluation — independent Validation/Test use, three training seeds, macro metrics, class-wise analysis and confidence intervals.

OOD detection and conformal prediction are outside the current main scope.

2. Important terminology

The current split uses GroupShuffleSplit(groups=case_num). Therefore, the repository describes it as a case-level grouped split. It should only be called a patient-level split if Derm7pt documentation independently confirms that case_num uniquely identifies patients.

Ground-truth concept substitution is reported as oracle concept intervention, not as a real doctor study.

3. Experimental models

B1_Clinical_Only

B2_Derm_Only

B3_Meta_Only

B4_Dual_NoMeta

B5_Dual_Metadata

B6_PureCBM

Proposed_Hybrid

Each final model is trained with seeds 42, 100, and 2026.

4. Protocol lock

The paper-ready protocol uses:

AdamW;

weighted Cross-Entropy for disease classification;

weighted BCEWithLogitsLoss for concepts;

explicit concept-loss coefficient alpha;

checkpoint selection by Validation Disease Macro-F1;

early stopping using Validation only;

Test used only after model/hyperparameter selection is locked;

mean ± sample standard deviation across independent seeds.

run_alpha_ablation.py writes the selected coefficient to outputs/selected_alpha.json. train_final_hybrid.py reads that file automatically.

5. Reproducibility pipeline

Run from the repository root.

pip install -r requirements.txt

Step 1 — Prepare data and metadata encoder

python -m src.data.prepare_data

Step 2 — Audit split distribution and case overlap

python -m src.check_distribution

Inspect results/split_audit.md before training.

Step 3 — Final alpha confirmation on Validation only

python -m src.run_alpha_ablation

This compares the final alpha candidates under the same optimizer, weighted losses, early stopping and checkpoint-selection protocol used for final training.

Step 4 — Train baselines B1–B6

python -m src.run_ablation

Step 5 — Train the final Proposed Hybrid model

python -m src.train_final_hybrid

Step 6 — Run independent Test evaluation

python -m src.run_evaluation

Outputs:

results/final_results.md

results/per_class_results.md

aggregated normalized confusion matrices in outputs/

Step 7 — Evaluate concept prediction quality

python -m src.concept_evaluation

Output:

results/concept_metrics.md

Step 8 — Concept-dependence analyses

python -m src.run_intervention
python -m src.sequential_cbm

These are oracle analyses, not prospective clinical-user studies.

Step 9 — Bootstrap confidence intervals

python -m src.bootstrap_eval

Output:

results/bootstrap_ci.md

Step 10 — Grad-CAM

python -m src.gradcam_vis

Use a seed chosen a priori or by Validation protocol; do not choose a visually favorable seed based on Test performance.

6. How to report current comparisons

Do not claim that the Proposed Hybrid model is universally superior unless the regenerated final table supports that statement across the chosen primary metric. Report each metric faithfully. In particular, distinguish:

the primary classification metric (recommended: Macro-F1 for imbalanced five-class classification);

Accuracy and AUROC as complementary metrics;

concept prediction metrics separately from disease metrics.

7. Repository outputs that should be committed

Commit aggregate, non-patient-level reports such as:

results/alpha_ablation_final.md

results/split_audit.md

results/final_results.md

results/per_class_results.md

results/concept_metrics.md

results/intervention_results.md

results/sequential_cbm_results.md

results/bootstrap_ci.md

Do not commit raw patient/image data or model checkpoints if licensing/privacy/storage constraints prohibit it.