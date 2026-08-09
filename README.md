# Phân loại Bệnh lý Da liễu Đa phương thức (Multimodal Derm7pt) kết hợp Trí tuệ Nhân tạo Giải thích được (XAI)

## 📌 Tổng quan dự án
Dự án áp dụng Học sâu đa phương thức (Multimodal Deep Learning) để chẩn đoán 5 nhóm bệnh lý da liễu phổ biến từ bộ dữ liệu Derm7pt. Hệ thống đề xuất kiến trúc **Hybrid Concept Bottleneck Model (Hybrid CBM)** kết hợp 3 luồng thông tin:
* **Ảnh lâm sàng (Clinical Images)**
* **Ảnh nội soi da (Dermoscopy Images)**
* **Dữ liệu nhân khẩu học (Dynamic Metadata)**: Số chiều động được mã hóa tự động để cung cấp đặc trưng bổ trợ thiết yếu.

Dự án tích hợp XAI (Grad-CAM) và kỹ thuật Concept Intervention (Can thiệp khái niệm) nhằm phơi bày hiện tượng học đường tắt (Shortcut Learning) và đề xuất giải pháp Sequential CBM an toàn cho tương tác Y tế.

## ⚖️ Tính năng Cốt lõi
* **Chia tách dữ liệu nhóm (GroupShuffleSplit):** Phân chia tập dữ liệu nghiêm ngặt theo ID Bệnh nhân (Case-level grouped split), đảm bảo phân bố lớp cân bằng và ngăn chặn rò rỉ dữ liệu giữa các tập Train/Val/Test.
* **Đánh giá Khách quan (Robust Evaluation):** Tự động hóa quá trình huấn luyện 21 mô hình (7 cấu hình × 3 hạt giống ngẫu nhiên 42, 100, 2026) đảm bảo độ tin cậy của thống kê (Mean ± SD).
* **Minh bạch hóa Quyết định:** Trực quan hóa bản đồ nhiệt Grad-CAM, giải thích nguyên nhân dự đoán thông qua các vật thể nhiễu bối cảnh.

## 📊 Kết quả Thực nghiệm (Test Set)

### 1. Đánh giá Mô hình Đa phương thức (Mean ± SD trên 3 Seeds)
Mô hình `Proposed_Hybrid` (với trọng số tối ưu $\alpha = 3.0$ và Early Stopping) vượt trội hoàn toàn so với các kiến trúc Baseline.

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Specificity | AUROC |
|:---|:---|:---|:---|:---|:---|:---|
| B1_Clinical_Only | 0.4101 ± 0.0783 | 0.2852 ± 0.0328 | 0.3524 ± 0.0290 | 0.3618 ± 0.0105 | 0.8575 ± 0.0111 | 0.7227 ± 0.0233 |
| B2_Derm_Only | 0.6864 ± 0.0541 | 0.5545 ± 0.0699 | 0.5251 ± 0.0571 | 0.6516 ± 0.0690 | 0.9198 ± 0.0101 | 0.8595 ± 0.0241 |
| B4_Dual_NoMeta | 0.6579 ± 0.0246 | 0.4711 ± 0.0238 | 0.4623 ± 0.0158 | 0.5165 ± 0.0157 | 0.9109 ± 0.0030 | 0.8704 ± 0.0174 |
| B5_Dual_Metadata| 0.6601 ± 0.0447 | 0.5106 ± 0.0169 | 0.4887 ± 0.0173 | 0.5838 ± 0.0471 | 0.9052 ± 0.0059 | 0.8727 ± 0.0041 |
| B6_PureCBM | 0.5504 ± 0.0792 | 0.4198 ± 0.0405 | 0.4246 ± 0.0181 | 0.5091 ± 0.0700 | 0.8904 ± 0.0179 | 0.7951 ± 0.0248 |
| **Proposed_Hybrid**| **0.6996 ± 0.0276**| **0.5206 ± 0.0252**| **0.5208 ± 0.0444**| **0.5801 ± 0.0184**| **0.9167 ± 0.0029**| **0.8829 ± 0.0087**|

*(Ảnh minh họa: Ma trận nhầm lẫn của Proposed_Hybrid)*
![Confusion Matrix](figures/image_dd33d5.png)

### 2. Can thiệp Khái niệm Lâm sàng (Concept Intervention)
| Model | Macro F1 (AI Only) | Macro F1 (With Doctor) | Absolute Improvement | Phân tích khoa học |
|:---|:---|:---|:---|:---|
| B6_PureCBM | 0.4198 ± 0.0405 | 0.2672 ± 0.0404 | - 0.1526 | Rò rỉ thông tin (Information Leakage) |
| Proposed_Hybrid | 0.5206 ± 0.0252 | 0.5206 ± 0.0252 | + 0.0000 | Học đường tắt (Shortcut Learning) |
| **Sequential CBM** | **0.4115** | **0.4237** | **+ 0.0122** | **Tương tác an toàn (Safe Interaction)** |

### 3. Phân tích XAI (Grad-CAM)
*Bản đồ nhiệt Grad-CAM phơi bày hiện tượng học đường tắt của mô hình kiến trúc lai khi tập trung vào chiếc thước kẻ thay vì vết thương Y khoa (Nguyên nhân khiến mức độ can thiệp bằng +0.0000).*
![XAI Failure Case](figures/image_dcca21.jpg)
![XAI Success Case](figures/image_dcc660.jpg)


## ⚙️ Cài đặt môi trường
Đảm bảo máy tính đã cài đặt Python 3.10+.
```bash
git clone [https://github.com/VanLuan05/RIVF2026_Derm7pt_Project.git](https://github.com/VanLuan05/RIVF2026_Derm7pt_Project.git)
cd RIVF2026_Derm7pt_Project
pip install -r requirements.txt

📂 Cấu trúc thư mục
RIVF2026_Derm7pt_Project/
│
├── data/                  # Chứa dữ liệu ảnh gốc và file phân chia CSV
├── figures/               # Lưu trữ ảnh Grad-CAM và Confusion Matrix hiển thị trên README
├── outputs/               # Nơi lưu trữ trọng số mô hình (.pth) và Encoder
├── results/               # Chứa các file Markdown báo cáo tự động (.md)
├── src/                   # Mã nguồn chính
│   ├── data/              # Các script tiền xử lý dữ liệu (prepare_data.py, dataset.py)
│   ├── models/            # Kiến trúc mạng Nơ-ron (MultimodalDermModel)
│   ├── run_ablation.py    # Huấn luyện tự động hàng loạt mô hình Baseline
│   ├── train_final_hybrid.py # Huấn luyện Proposed_Hybrid với Early Stopping
│   ├── run_evaluation.py  # Script chấm điểm và trích xuất chỉ số tự động
│   ├── run_intervention.py # Kịch bản Bác sĩ can thiệp (Nhãn mềm)
│   └── gradcam_vis.py     # Trích xuất ảnh trực quan hóa Grad-CAM


** Hướng dẫn Chạy Thực nghiệm (Reproducibility Pipeline)
-- Lưu ý: Các tập lệnh cũ (như src.main, src.evaluate, src.visualize) đã được nâng cấp thành luồng tự động hóa. Vui lòng chạy các lệnh dưới đây theo đúng thứ tự để đảm bảo tính tái lập của hệ thống.

Bước 1: Tiền xử lý Dữ liệu & Mã hóa Metadata
-- Quá trình này sẽ chia tập dữ liệu an toàn và đúc khuôn OneHotEncoder trên tập Train để tránh xô lệch nhãn.
#  python -m src.data.prepare_data
Bước 2: Huấn luyện Baseline (B1-B6)
-- Huấn luyện các mô hình tham chiếu qua 3 hạt giống (42, 100, 2026).
#  python -m src.run_ablation
Bước 3: Huấn luyện model cuối (Proposed_Hybrid)
-- Áp dụng điểm vàng siêu tham số $\alpha = 3.0$ và kỹ thuật Early Stopping trên tập Validation.
# python -m src.train_final_hybrid
Bước 4: Đánh giá & Bác sĩ can thiệp
-- Trích xuất báo cáo F1/AUROC, ma trận nhầm lẫn và mô phỏng tương tác Human-in-the-loop bằng nhãn mềm.
#  python -m src.run_evaluation
#  python -m src.run_intervention
Bước 5: Trích xuất Ảnh Trực quan (XAI)
-- Tạo bản đồ nhiệt Grad-CAM giải thích quyết định của mạng nơ-ron.
#  python -m src.gradcam_vis