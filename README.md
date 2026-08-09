# Phân loại Bệnh lý Da liễu Đa phương thức (Multimodal Derm7pt) kết hợp Trí tuệ Nhân tạo Giải thích được (XAI)


## Tổng quan dự án
Dự án áp dụng Học sâu đa phương thức (Multimodal Deep Learning) để chẩn đoán 5 loại bệnh lý da liễu phổ biến từ bộ dữ liệu Derm7pt. Hệ thống sử dụng kiến trúc lai (Hybrid/Multitask) kết hợp 3 luồng thông tin:
* **Ảnh lâm sàng (Clinical Images)**
* **Ảnh nội soi da (Dermoscopy Images)**
* **Dữ liệu nhân khẩu học (Metadata)**: Được chuẩn hóa nghiêm ngặt qua 14 chiều dữ liệu để ngăn chặn nhiễu.

Dự án tích hợp các phương pháp Trí tuệ Nhân tạo Giải thích được (Explainable AI - XAI) thông qua **Grad-CAM** và kỹ thuật **Concept Intervention** (Can thiệp khái niệm lâm sàng) nhằm nâng cao độ tin cậy trong y tế.

## Tính năng Cốt lõi (MLOps Standard)
* **Chia tách dữ liệu an toàn (Data Splitting):** Phân chia tập dữ liệu nghiêm ngặt theo ID Bệnh nhân (Case-level grouped split) để ngăn chặn tuyệt đối hiện tượng rò rỉ dữ liệu (Data Leakage - Chuẩn P0).
* **Mô hình Khái niệm Chuỗi (Sequential CBM):** Hỗ trợ Bác sĩ can thiệp (human-in-the-loop) vào các khái niệm lâm sàng một cách an toàn để cải thiện độ chính xác chẩn đoán.
* **Đánh giá Khách quan (Robust Evaluation):** Tự động hóa quá trình huấn luyện và kiểm thử với 3 hạt giống ngẫu nhiên (3-seed automation) để đảm bảo độ tin cậy của các chỉ số thống kê (Mean ± SD).

## Cài đặt môi trường
Đảm bảo máy tính của bạn đã cài đặt Python 3.10+. Khởi tạo môi trường ảo và chạy lệnh sau để tải mã nguồn và cài đặt thư viện:
```bash
git clone [https://github.com/VanLuan05/RIVF2026_Derm7pt_Project.git](https://github.com/VanLuan05/RIVF2026_Derm7pt_Project.git)
cd RIVF2026_Derm7pt_Project
pip install -r requirements.txt

📂 Cấu trúc thư mục
RIVF2026_Derm7pt_Project/
│
├── data/                  # Chứa dữ liệu ảnh gốc và file phân chia CSV
├── outputs/               # Nơi lưu trữ trọng số mô hình (.pth), file CSV và biểu đồ XAI
├── src/                   # Mã nguồn chính
│   ├── data/              # Các script tiền xử lý dữ liệu và DataLoader (prepare_data.py, dataset.py)
│   ├── models/            # Kiến trúc mạng Nơ-ron (MultimodalDermModel)
│   ├── run_ablation.py    # Kịch bản tự động huấn luyện hàng loạt (B1-B5, P2)
│   ├── run_evaluation.py  # Script chấm điểm và trích xuất chỉ số tự động
│   ├── sequential_cbm.py  # Kịch bản Bác sĩ can thiệp (Sequential CBM)
│   └── gradcam_vis.py     # Trích xuất ảnh trực quan hóa Grad-CAM

** Hướng dẫn Chạy Thực nghiệm (Reproducibility Pipeline)
-- Lưu ý: Các tập lệnh cũ (như src.main, src.evaluate, src.visualize) đã được nâng cấp thành luồng tự động hóa. Vui lòng chạy các lệnh dưới đây theo đúng thứ tự để đảm bảo tính tái lập của hệ thống.

Bước 1: Tiền xử lý Dữ liệu & Mã hóa Metadata
-- Quá trình này sẽ chia tập dữ liệu an toàn và đúc khuôn OneHotEncoder trên tập Train để tránh xô lệch nhãn.
#  python -m src.data.prepare_data
Bước 2: Huấn luyện Tự động Hàng loạt (Ablation Training)
-- Tự động huấn luyện các mô hình baseline từ B1 đến B5 và mô hình Master_P2 qua 3 seed ngẫu nhiên (42, 100, 2026).
#  python -m src.run_ablation
Bước 3: Đánh giá & Trích xuất Chỉ số (Evaluation)
-- Chấm điểm 21 phiên bản (7 cấu hình × 3 seeds) mô hình trên tập Test độc lập và tự động xuất báo cáo tổng hợp ra file outputs/final_results_summary.csv.
#  python -m src.run_evaluation
Bước 4: Bác sĩ Can thiệp Lâm sàng (Sequential CBM)
-- Đo lường sự gia tăng F1-Score khi có sự hợp tác giữa AI và Bác sĩ thông qua mô hình chuỗi độc lập.
#  python -m src.sequential_cbm
Bước 5: Trích xuất Ảnh Trực quan Grad-CAM (XAI)
-- Quét qua tập Test để tạo bản đồ nhiệt (Heatmap) trực quan hóa, trích xuất các ca AI dự đoán đúng (Success) và dự đoán sai (Failure).
#  python -m src.gradcam_vis