# RIVF2026 Derm7pt — Mô hình Hybrid Concept Bottleneck Đa phương thức

## 1. Phạm vi nghiên cứu

Đề tài nghiên cứu bài toán **phân loại tổn thương da thành 5 lớp** trên bộ dữ liệu **Derm7pt**, sử dụng:

* ảnh lâm sàng (clinical images);
* ảnh soi da (dermoscopy images);
* dữ liệu siêu dữ liệu có cấu trúc (structured metadata);
* bảy khái niệm có ý nghĩa lâm sàng của Derm7pt;
* **Pure Concept Bottleneck Model (Pure CBM)**;
* **Hybrid Concept Bottleneck Model (Hybrid CBM)** được đề xuất;
* **Grad-CAM** và phân tích **oracle concept** nhằm tăng khả năng giải thích mô hình.

Phạm vi của bài báo được giới hạn có chủ đích vào **ba đóng góp chính**:

### 1. Học đa phương thức với metadata

Định lượng giá trị đóng góp của ba nguồn dữ liệu:

* ảnh lâm sàng;
* ảnh soi da;
* metadata.

### 2. Mô hình hóa dựa trên concept

So sánh giữa:

* các mô hình đa phương thức dạng **black-box**;
* **Pure CBM**;
* **Proposed Hybrid CBM**.

### 3. Thực nghiệm và đánh giá nghiêm ngặt

Áp dụng:

* lựa chọn mô hình/hyperparameter **chỉ trên tập Validation**;
* ba seed huấn luyện;
* đánh giá độc lập trên tập Test;
* các chỉ số macro;
* phân tích theo từng lớp;
* khoảng tin cậy (confidence intervals).

**OOD detection** và **conformal prediction** chưa được thực hiện.

---
## Cấu trúc thực nghiệm của repository

Ba đóng góp chính của bài báo được triển khai trong repository thông qua 5 module thực nghiệm:

### C1 — Metadata-Guided Cross-Attention
Đánh giá cơ chế fusion trong đó metadata đóng vai trò Query, còn đặc trưng clinical và dermoscopy đóng vai trò Key/Value.

C1 là một kiến trúc fusion thay thế, không được khẳng định là luôn tốt hơn hoặc nhanh hơn feature concatenation.

### C2 — Concept Bottleneck & Concept Intervention
Bao gồm Pure CBM, Proposed Hybrid CBM, concept prediction, oracle intervention và sequential concept diagnostics.

Alpha được chọn trên Validation với `alpha = 2.0`.

### C3 — Rigorous Experimental Evaluation
Bao gồm 3 seed `42`, `100`, `2026`, lựa chọn mô hình trên Validation, đánh giá độc lập trên Test, macro metrics, per-class analysis và bootstrap CI.

### C4 — Visual Explainability with Grad-CAM
Grad-CAM được dùng để phân tích định tính vùng ảnh ảnh hưởng đến prediction trên clinical và dermoscopy.

Phân tích dùng `Proposed_Hybrid`, seed cố định `42`, gồm 5 Success và 5 Failure cases đại diện đủ 5 lớp.

### C5 — Prediction Confidence & Uncertainty Analysis
Bao gồm confidence, entropy, prediction margin, seed disagreement, probability variability, mutual information, error detection, uncertainty ranking và selective prediction.

C5 không thực hiện OOD detection, conformal prediction hoặc prospective clinical triage.

Các báo cáo canonical nằm tại:

```text
results/contribution_1/
results/contribution_2/
results/contribution_3/
results/contribution_4/
results/contribution_5/




`GroupShuffleSplit(groups=case_num)`

Do đó, repository mô tả cách chia dữ liệu là **chia theo nhóm ở cấp case (case-level grouped split)**.

Chỉ được gọi đây là **patient-level split (chia ở cấp bệnh nhân)** nếu tài liệu chính thức của Derm7pt xác nhận độc lập rằng `case_num` xác định duy nhất từng bệnh nhân.

Việc thay thế concept bằng **ground-truth concept** được báo cáo dưới dạng:

> **oracle concept analysis/intervention**

hay **phân tích/can thiệp concept oracle**, chứ **không được mô tả là một nghiên cứu thực tế với bác sĩ lâm sàng**.

---

# 3. Các mô hình thực nghiệm cuối cùng

File `run_ablation.py` huấn luyện toàn bộ **7 kiến trúc**:

| Mã               | Kiến trúc                                    |
| ---------------- | -------------------------------------------- |
| B1_Clinical_Only | Chỉ sử dụng ảnh lâm sàng                     |
| B2_Derm_Only     | Chỉ sử dụng ảnh soi da                       |
| B3_Meta_Only     | Chỉ sử dụng metadata                         |
| B4_Dual_NoMeta   | Hai loại ảnh, không sử dụng metadata         |
| B5_Dual_Metadata | Hai loại ảnh + metadata                      |
| B6_PureCBM       | Pure Concept Bottleneck Model                |
| Proposed_Hybrid  | Hybrid Concept Bottleneck Model được đề xuất |

Mỗi kiến trúc được huấn luyện với **3 seed**:

* `42`
* `100`
* `2026`

Tổng cộng:

**7 kiến trúc × 3 seed = 21 checkpoint** phục vụ bài báo.

Không có bước `train_final_hybrid.py` riêng trong quy trình cuối cùng.

---

# 4. Quy trình huấn luyện được cố định

Tất cả các mô hình được sử dụng trong bài báo đều tuân theo cùng một quy trình cốt lõi:

* **AdamW**;
* learning rate = `5e-5`;
* weight decay = `1e-4`;
* **Weighted Cross-Entropy** cho bài toán phân loại bệnh;
* **Weighted BCEWithLogitsLoss** cho concept khi mô hình có concept head;
* sử dụng hệ số trọng số loss của concept `alpha` một cách tường minh;
* lựa chọn checkpoint dựa trên **Validation Disease Macro-F1**;
* **early stopping chỉ sử dụng tập Validation**;
* chỉ sử dụng tập Test **sau khi toàn bộ protocol đã được cố định**;
* kết quả được báo cáo dưới dạng **mean ± sample SD** trên ba seed.

File:

`run_alpha_ablation.py`

sẽ lựa chọn giá trị `alpha` cuối cùng **chỉ dựa trên Validation** và ghi kết quả vào:

```text
outputs/selected_alpha.json
```

File `run_ablation.py` sẽ **không cho phép bắt đầu huấn luyện nếu file này chưa tồn tại**.

---

# 5. Chính sách lưu trữ trên Google Colab

Để tăng tốc độ huấn luyện, ảnh được sao chép từ Google Drive sang bộ nhớ local của Colab **một lần cho mỗi phiên làm việc**:

```bash
rm -rf /content/local_images
cp -a /content/drive/MyDrive/RIVF2026_Dataset/data/raw/images /content/local_images
```

`Config.runtime_paths()` sẽ ưu tiên:

```text
/content/local_images
```

nếu thư mục này tồn tại.

Nếu không tồn tại, chương trình sẽ sử dụng đường dẫn dự phòng:

```text
data/raw/images
```

Để bảo toàn checkpoint và kết quả sau khi Colab reset, các thư mục trong repository được liên kết symbolic link với Google Drive:

```python
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
```

---
📂 Cấu trúc thư mục
RIVF2026_Derm7pt_Project/
│
├── data/                               # Thư mục chứa dữ liệu (Không push lên GitHub)
│   ├── raw/
│   │   └── images/                     # Ảnh gốc Derm7pt
│   └── processed/                      # Dữ liệu đã qua xử lý
│       ├── train_split.csv             # Tập Train
│       ├── val_split.csv               # Tập Validation (Dùng để chọn Alpha và Checkpoint)
│       ├── calib_split.csv             # Tập Calibration
│       ├── test_split.csv              # Tập Test (Chỉ dùng đánh giá cuối cùng)
│       └── label_mapping.json          # File ánh xạ nhãn bệnh và khái niệm
│
├── notebooks/                          # Chứa các file Jupyter Notebook phác thảo
│   └── 01_eda_and_split.ipynb          # Notebook EDA và chia tập dữ liệu ban đầu
│
├── outputs/                            # Nơi lưu trữ Model và các Artifacts sinh ra khi train
│   ├── meta_encoder.joblib             # File OneHotEncoder mã hóa Metadata
│   ├── selected_alpha.json             # File lưu giá trị Alpha tốt nhất (được chọn từ validation)
│   ├── gradcam_results/                # Chứa các ảnh giải thích Grad-CAM (Success & Failure)
│   └── *.pth                           # 21 Checkpoints của các models (B1-B6, Proposed_Hybrid x 3 seeds)
│
├── results/                            # Chứa các báo cáo kết quả đánh giá (Đẩy lên GitHub)
│   ├── split_audit.md                  # Báo cáo kiểm tra rò rỉ dữ liệu
│   ├── ablation_training_manifest.json # File log ghi nhận hoàn tất train 21 models
│   ├── final_results.md                # Báo cáo điểm số Test set (Accuracy, F1, AUROC...)
│   ├── concept_metrics.md              # Báo cáo dự đoán 7 Khái niệm Y khoa
│   ├── intervention_results.md         # Phân tích Oracle Concept Intervention
│   ├── sequential_cbm_results.md       # Phân tích Sequential CBM (AI vs Oracle)
│   └── test_predictions.csv            # File chứa log dự đoán, xác suất và độ tự tin (phục vụ Phase 2 & 3)
│
├── src/                                # Thư mục chứa MÃ NGUỒN CHÍNH
│   ├── config.py                       # File cấu hình trung tâm (Đường dẫn động, Siêu tham số)
│   ├── train.py                        # Cốt lõi huấn luyện (P0: truyền alpha, chọn checkpoint bằng F1)
│   ├── check_distribution.py           # Script kiểm toán rò rỉ dữ liệu bệnh nhân
│   ├── run_alpha_ablation.py           # Script tìm Alpha tốt nhất trên tập Validation
│   ├── run_ablation.py                 # Kịch bản tự động train 21 models
│   ├── run_evaluation.py               # Chấm điểm trên tập Test (xuất test_predictions.csv)
│   ├── concept_evaluation.py            # Chấm điểm 7 khái niệm (AUROC, F1, Precision, Recall)
│   ├── run_intervention.py             # Đánh giá thay thế Oracle Concept trực tiếp
│   ├── sequential_cbm.py               # Đánh giá CBM chuỗi qua 3 seeds (Random Forest/Logistic Regression)
│   ├── bootstrap_eval.py               # Trích xuất khoảng tin cậy 95% Bootstrap CI
│   ├── gradcam_vis.py                  # Vẽ bản đồ nhiệt Grad-CAM
│   ├── extract_predictions.py          # Trích xuất probability, confidence và uncertainty trên Test
│   │
│   ├── data/                           # Module Xử lý Dữ liệu
│   │   ├── dataset.py                  # Class PyTorch Dataset và Transform
│   │   └── prepare_data.py             # Script tiền xử lý dữ liệu và chia split
│   │
│   └── models/                         # Module Kiến trúc Mạng Nơ-ron
│       └── models.py                   # Định nghĩa MultimodalDermModel (ResNet, Concept Bottleneck, Fusion)
│
├── requirements.txt                    # Danh sách thư viện Python cần thiết
├── .gitignore                          # Cấu hình bỏ qua các file nặng/nhạy cảm khi push Git
└── README.md                           # Tóm tắt dự án, hướng dẫn chạy và kết quả học thuật


# 6. Quy trình thực thi cuối cùng cho bài báo

Tất cả lệnh phải được chạy từ **thư mục gốc của repository**.

Không được bỏ qua các bước **audit** và **alpha gate**.

## Bước 1 — Chuẩn bị dữ liệu

```bash
python -m src.data.prepare_data
```

Lệnh này tạo:

* các tập Train/Validation/Calibration/Test đã được xử lý;
* ánh xạ nhãn chung (common label mapping);
* bộ mã hóa metadata (metadata encoder).

---

## Bước 2 — Kiểm tra việc chia dữ liệu và tham chiếu ảnh

```bash
python -m src.check_distribution
```

Kiểm tra file:

```text
results/split_audit.md
```

**Không được bắt đầu huấn luyện nếu lệnh này phát sinh lỗi.**

---

## Bước 3 — Xác nhận alpha chỉ trên Validation

```bash
python -m src.run_alpha_ablation
```

Kiểm tra:

```bash
cat outputs/selected_alpha.json
cat results/alpha_ablation_final.md
```

**Không sử dụng bất kỳ Test metric nào để lựa chọn alpha.**

---

## Bước 4 — Huấn luyện toàn bộ 21 mô hình

```bash
python -m src.run_ablation
```

Các checkpoint dự kiến:

```text
B1_Clinical_Only_seed_42.pth
...
B6_PureCBM_seed_2026.pth
Proposed_Hybrid_seed_42.pth
Proposed_Hybrid_seed_100.pth
Proposed_Hybrid_seed_2026.pth
```

Manifest của quá trình huấn luyện được lưu tại:

```text
results/ablation_training_manifest.json
```

---

## Bước 5 — Đánh giá độc lập trên Test

```bash
python -m src.run_evaluation
```

Kết quả chính:

```text
results/final_results.md
```

---

## Bước 6 — Đánh giá khả năng dự đoán concept

```bash
python -m src.concept_evaluation
```

Kết quả chính:

```text
results/concept_metrics.md
```

---

## Bước 7 — Thay thế concept oracle trực tiếp

```bash
python -m src.run_intervention
```

Kết quả chính:

```text
results/intervention_results.md
```

---

## Bước 8 — Phân tích khoảng cách chất lượng concept của Sequential CBM

```bash
python -m src.sequential_cbm
```

Kết quả chính:

```text
results/sequential_cbm_results.md
```

---

## Bước 9 — Khoảng tin cậy bằng Bootstrap

```bash
python -m src.bootstrap_eval
```

Kết quả chính:

```text
results/bootstrap_ci.md
```

---

## Bước 10 — Grad-CAM

```bash
python -m src.gradcam_vis
```

Các hình ảnh được lưu tại:

```text
outputs/gradcam_results/
```

---

# 7. Quy tắc diễn giải kết quả trong bài báo

### Không được khẳng định Proposed Hybrid tốt hơn một cách tuyệt đối

Không được tuyên bố rằng **Proposed Hybrid** luôn vượt trội mọi mô hình khác, trừ khi bảng kết quả **Test cuối cùng** thực sự chứng minh điều đó đối với **primary metric** đã được xác định.

### Không được lựa chọn seed dựa trên Test

Không được chọn seed có kết quả Test tốt nhất để báo cáo như kết quả chính.

### Không được tuning bằng Test

Không được sử dụng Test để điều chỉnh:

* `alpha`;
* hyperparameter;
* hoặc bất kỳ quyết định lựa chọn mô hình nào.

### Không gọi case-level split là patient-level split

Không được mô tả:

```text
case_num grouping
```

là **patient-level split** nếu chưa có xác nhận độc lập rằng `case_num` xác định duy nhất bệnh nhân.

### Không gọi oracle intervention là nghiên cứu với bác sĩ

Việc thay thế concept bằng ground-truth concept phải được gọi là:

> **oracle concept analysis/intervention**

Không được mô tả đây là một **can thiệp thực tế của bác sĩ**.

### Báo cáo độ không chắc chắn chính

Kết quả chính về độ không chắc chắn của quá trình huấn luyện là:

> **Mean ± Sample SD trên 3 seed**

Việc sử dụng **probability ensemble của ba seed trong `bootstrap_eval.py`** chỉ được xem là **phân tích thứ cấp (secondary analysis)**.

---

# 8. Môi trường thực thi

Cài đặt các thư viện cần thiết bằng:

```bash
pip install -r requirements.txt
```

Repository yêu cầu các thư viện:

* PyTorch;
* torchvision;
* NumPy;
* pandas;
* scikit-learn;
* Pillow;
* matplotlib;
* seaborn;
* joblib;
* tqdm;
* tabulate;
* grad-cam.
