RIVF2026 Derm7pt — Multimodal Hybrid Concept Bottleneck Model

1. Research scope

This project studies five-class skin-lesion classification on Derm7pt using:

clinical images;

dermoscopy images;

structured metadata;

seven clinically motivated Derm7pt concepts;

a Pure Concept Bottleneck Model (Pure CBM);

a proposed Hybrid Concept Bottleneck Model (Hybrid CBM);

Grad-CAM and oracle concept analyses for interpretability.

The paper-ready scope is intentionally limited to three main contributions:

Multimodal learning with metadata — quantify the value of clinical, dermoscopic and metadata streams.

Concept-based modeling — compare black-box multimodal baselines, Pure CBM and Proposed Hybrid CBM.

Rigorous ablation/evaluation — Validation-only model/hyperparameter selection, three training seeds, independent Test evaluation, macro metrics, class-wise analysis and confidence intervals.

OOD detection and conformal prediction are outside the current main scope.

2. Important terminology

The current data preparation uses GroupShuffleSplit(groups=case_num). Therefore, this repository describes the split as a case-level grouped split. It should only be called a patient-level split if Derm7pt documentation independently confirms that case_num uniquely identifies patients.

Ground-truth concept substitution is reported as oracle concept analysis/intervention, not as a real clinician study.

3. Final experimental models

run_ablation.py trains all seven architectures:

B1_Clinical_Only

B2_Derm_Only

B3_Meta_Only

B4_Dual_NoMeta

B5_Dual_Metadata

B6_PureCBM

Proposed_Hybrid

Each architecture is trained with seeds 42, 100, and 2026.

Total: 7 architectures × 3 seeds = 21 paper checkpoints.

There is no separate train_final_hybrid.py step in the final workflow.

4. Locked training protocol

All paper models use the same core protocol:

AdamW;

learning rate 5e-5;

weight decay 1e-4;

weighted Cross-Entropy for disease classification;

weighted BCEWithLogitsLoss for concepts when a concept head exists;

explicit concept-loss coefficient alpha;

checkpoint selection by Validation Disease Macro-F1;

early stopping using Validation only;

Test used only after the protocol is locked;

results reported as mean ± sample SD across three seeds.

run_alpha_ablation.py chooses the final alpha using Validation only and writes:

outputs/selected_alpha.json

run_ablation.py refuses to start without this file.

5. Colab storage policy

For faster training, copy images from Drive to Colab local storage once per session:

rm -rf /content/local_images
cp -a /content/drive/MyDrive/RIVF2026_Dataset/data/raw/images /content/local_images

Config.runtime_paths() prefers /content/local_images when it exists. Otherwise it falls back to data/raw/images.

To preserve checkpoints/results across Colab resets, symlink repository folders to Google Drive:

import os

PERSIST = "/content/drive/MyDrive/RIVF2026_Dataset"
os.makedirs(f"{PERSIST}/outputs", exist_ok=True)
os.makedirs(f"{PERSIST}/results", exist_ok=True)

%cd /content/RIVF2026_Derm7pt_Project

!cp -a outputs/. "$PERSIST/outputs/" 2>/dev/null || true
!cp -a results/. "$PERSIST/results/" 2>/dev/null || true

!rm -rf outputs results
!ln -s "$PERSIST/outputs" outputs
!ln -s "$PERSIST/results" results

6. Paper-final execution order

Run from the repository root. Do not skip the audit/alpha gates.

Step 1 — Prepare data

python -m src.data.prepare_data

This creates the processed Train/Validation/Calibration/Test splits, the common label mapping, and the metadata encoder.

Step 2 — Audit the split and image references

python -m src.check_distribution

Inspect:

results/split_audit.md

Training should not start if this command raises an error.

Step 3 — Confirm alpha on Validation only

python -m src.run_alpha_ablation

Inspect:

cat outputs/selected_alpha.json
cat results/alpha_ablation_final.md

No Test metric is used to select alpha.

Step 4 — Train all 21 paper models

python -m src.run_ablation

Expected checkpoint pattern:

B1_Clinical_Only_seed_42.pth
...
B6_PureCBM_seed_2026.pth
Proposed_Hybrid_seed_42.pth
Proposed_Hybrid_seed_100.pth
Proposed_Hybrid_seed_2026.pth

The training manifest is saved to:

results/ablation_training_manifest.json

Step 5 — Independent Test evaluation

python -m src.run_evaluation

Main result:

results/final_results.md

Step 6 — Concept prediction evaluation

python -m src.concept_evaluation

Main result:

results/concept_metrics.md

Step 7 — Direct oracle concept substitution

python -m src.run_intervention

Main result:

results/intervention_results.md

Step 8 — Sequential CBM concept-quality gap analysis

python -m src.sequential_cbm

Main result:

results/sequential_cbm_results.md

Step 9 — Bootstrap confidence intervals

python -m src.bootstrap_eval

Main result:

results/bootstrap_ci.md

Step 10 — Grad-CAM

python -m src.gradcam_vis

Figures are written to:

outputs/gradcam_results/

7. Interpretation rules for the paper

Do not claim the Proposed Hybrid model is universally superior unless the final Test table supports that statement for the stated primary metric.

Do not select a seed using Test performance.

Do not tune alpha or other hyperparameters using Test.

Do not describe case_num grouping as patient-level without independent confirmation.

Do not describe ground-truth concept substitution as a real doctor intervention.

Keep the three-seed mean ± sample SD as the primary training-run uncertainty report.

Treat the three-seed probability ensemble in bootstrap_eval.py as a secondary analysis.

8. Environment

Install dependencies with:

pip install -r requirements.txt

The repository expects PyTorch, torchvision, NumPy, pandas, scikit-learn, Pillow, matplotlib, seaborn, joblib, tqdm, tabulate and grad-cam.