# CONTRIBUTION 4 - FINAL SUMMARY

## 1. Mục tiêu

Xây dựng quy trình đánh giá nghiêm ngặt và có khả năng tái lập cho mô hình,
sử dụng các tập Train, Validation, Calibration và Test tách biệt,
kiểm soát data leakage, lựa chọn mô hình chỉ trên Validation,
đánh giá trên nhiều random seeds và định lượng độ bất định của kết quả.

## 2. Data Split Protocol

Tổng số mẫu: **1011**

| Split | Samples | Vai trò |
|---|---:|---|
| Train | 605 | Model fitting |
| Validation | 152 | Checkpoint & hyperparameter selection |
| Calibration | 102 | Reserved calibration set |
| Test | 152 | Final evaluation only |

Split strategy:
- Grouped splitting
- Primary grouping key: `case_num`
- Additional leakage check: `case_id`
- `case_num` overlap giữa các split: **0**
- `case_id` overlap giữa các split: **0**

Không mô tả protocol này là patient-level split vì `case_num`
chưa được xác nhận là patient identifier.

## 3. Leakage Control

Audit được thực hiện giữa mọi cặp:

- Train ↔ Validation
- Train ↔ Calibration
- Train ↔ Test
- Validation ↔ Calibration
- Validation ↔ Test
- Calibration ↔ Test

Kết quả:

**Không phát hiện overlap theo `case_num` hoặc `case_id`.**

Metadata preprocessing:
- Columns: `sex`, `location`, `elevation`
- Metadata encoder chỉ được fit trên **Train**
- Validation / Calibration / Test không tham gia fit encoder
- `handle_unknown='ignore'`

## 4. Model & Hyperparameter Selection

Hyperparameter alpha được lựa chọn hoàn toàn trên Validation.

Candidate alpha:
- 2.0
- 3.0

Selection metric:

`mean_validation_disease_macro_f1_across_3_seeds`

Selected alpha:

**alpha = 2.0**

Test set không được sử dụng để lựa chọn alpha.

Checkpoint selection:
- Monitor: Validation Disease Macro-F1
- Early stopping patience: 5
- Maximum epochs: 20

## 5. Reproducibility Protocol

Locked seeds:

**42, 100, 2026**

Training configuration:

| Setting | Value |
|---|---|
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 5e-05 |
| Weight decay | 0.0001 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Checkpoint monitor | Validation Disease Macro-F1 |

Randomness control:
- Python random seed
- NumPy seed
- PyTorch seed
- CUDA seed
- `cudnn.deterministic = True`
- `cudnn.benchmark = False`
- Seeded DataLoader generator

## 6. Multi-Seed Final Evaluation

Final evaluation được thực hiện trên independent Test split sau khi
model/hyperparameter selection đã hoàn tất trên Validation.

Có:

- **7 architectures**
- **3 seeds**
- **21 trained models**

Primary reporting:

**Mean ± sample standard deviation across seeds (`ddof=1`)**

Proposed Hybrid Test results:

| Metric | Mean ± SD |
|---|---:|
| Accuracy | 0.6842 ± 0.0287 |
| Balanced Accuracy | 0.5084 ± 0.0276 |
| Macro F1 | 0.4781 ± 0.0279 |
| Macro Precision | 0.4660 ± 0.0250 |
| Macro Recall | 0.5084 ± 0.0276 |
| Macro Specificity | 0.9133 ± 0.0072 |
| One-vs-Rest AUROC | 0.8555 ± 0.0217 |

## 7. Bootstrap Uncertainty Estimation

Method:

**Stratified percentile bootstrap**

Configuration:
- Bootstrap replicates: **1000**
- Random state: **42**
- Confidence interval: **95%**

Resampling được thực hiện trong từng disease class để giữ nguyên
class counts trong mỗi bootstrap replicate.

Per-seed Macro-F1 95% CI:

| Seed | Point Estimate | 95% CI |
|---|---:|---:|
| 42 | 0.5076 | [0.4022, 0.6246] |
| 100 | 0.4746 | [0.3819, 0.5780] |
| 2026 | 0.4521 | [0.3589, 0.5435] |

3-seed probability ensemble được xem là **secondary analysis**,
không thay thế primary reporting bằng mean ± SD across independent runs.

## 8. Vai trò của bốn tập dữ liệu

Train:
- Model fitting
- Fit metadata encoder

Validation:
- Checkpoint selection
- Early stopping
- Hyperparameter alpha selection

Calibration:
- Được giữ riêng
- Hiện tại chưa được sử dụng để model selection hay training
- Chỉ được mô tả là **reserved calibration set**

Test:
- Không dùng cho training
- Không dùng để chọn checkpoint
- Không dùng để chọn alpha
- Chỉ dùng cho final evaluation và post-training analyses

## 9. Kết luận C4

Contribution 4 đã triển khai một evaluation protocol có kiểm soát leakage
và có khả năng tái lập.

Protocol bao gồm:

- Train / Validation / Calibration / Test tách biệt
- Grouped split với zero overlap theo `case_num` và `case_id`
- Metadata encoder fit trên Train only
- Hyperparameter và checkpoint selection trên Validation only
- Independent Test evaluation
- Three locked random seeds
- Mean ± sample SD across seeds
- Stratified bootstrap 95% confidence intervals
- Fixed bootstrap random state

Vì Calibration set hiện mới được reserve và chưa được sử dụng cho calibration,
không claim rằng model đã được calibrated trong Contribution 4.

**Contribution 4: Rigorous and reproducible evaluation protocol — COMPLETED.**

## 10. Files liên quan

### C4 Evidence
- `results/contribution_4/c4_split_audit.csv`
- `results/contribution_4/c4_protocol.json`
- `results/contribution_4/c4_selected_alpha.json`
- `results/contribution_4/c4_alpha_ablation_final.md`
- `results/contribution_4/c4_final_results.md`
- `results/contribution_4/c4_bootstrap_ci.md`
- `results/contribution_4/c4_ablation_training_manifest.json`

### Main source
- `src/data/prepare_data.py`
- `src/config.py`
- `src/train.py`
- `src/run_ablation.py`
- `src/run_alpha_ablation.py`
- `src/run_evaluation.py`
- `src/bootstrap_eval.py`
