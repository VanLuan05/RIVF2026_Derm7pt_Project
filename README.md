# Dự án RIVF2026 - AI Đáng Tin Cậy cho Da Liễu
Dự án sử dụng bộ dữ liệu Derm7pt, tích hợp đa phương thức và nút thắt khái niệm (CBM).
# Multimodal Skin Lesion Classification (Derm7pt)

## Tổng quan dự án
Dự án áp dụng Học sâu đa phương thức (Multimodal Deep Learning) để chẩn đoán 5 loại bệnh lý da liễu phổ biến từ bộ dữ liệu Derm7pt. Hệ thống sử dụng kiến trúc lai (Hybrid/Multitask) kết hợp 3 luồng thông tin:
*   **Ảnh lâm sàng (Clinical Images)**
*   **Ảnh nội soi da (Dermoscopy Images)**
*   **Dữ liệu nhân khẩu học (Metadata)**

Dự án tích hợp các phương pháp Trí tuệ Nhân tạo Giải thích được (Explainable AI - XAI) thông qua **Grad-CAM** và kỹ thuật **Concept Intervention** (Can thiệp khái niệm lâm sàng) nhằm nâng cao độ tin cậy trong y tế.

## Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.10+. Khởi tạo môi trường ảo và chạy lệnh sau:
```bash
pip install -r requirements.txt

## Cấu trúc thư mục
RIVF2026_Derm7pt_Project/
│
├── data/                   # Chứa dữ liệu ảnh gốc và file phân chia CSV
├── outputs/                # Nơi lưu trữ trọng số mô hình (.pth) và biểu đồ
├── src/                    # Mã nguồn chính
│   ├── config.py           # Thiết lập đường dẫn và tham số toàn cục
│   ├── data/               # Các script tiền xử lý dữ liệu và DataLoader
│   ├── models/             # Kiến trúc mạng Nơ-ron (ResNet50, MetaEncoder)
│   ├── train.py            # Vòng lặp huấn luyện chính
│   ├── evaluate.py         # Script đánh giá và trích xuất chỉ số
│   ├── intervention.py     # Giả lập kịch bản Bác sĩ can thiệp khái niệm
│   ├── sequential_cbm.py   # Mô hình Chuỗi độc lập (Sequential CBM)
│   └── evaluate_robustness.py # Kiểm định tính bền vững (Robustness Test)

##---Hướng dẫn sử dụng---##
-- 1. Huấn luyện mô hình: Chạy tập lệnh chính để bắt đầu quá trình đào tạo với các Seed khác nhau:
---------- python -m src.main
-- 2. Đánh giá và xuất báo cáo:Lấy kết quả Ma trận nhầm lẫn (Confusion Matrix) và Đường cong ROC:
---------- python -m src.evaluate
---------- python -m src.visualize
-- 3. Kiểm tra tính ổn định (Robustness Test): Chạy suy luận trên tập Test độc lập để lấy Khoảng tin cậy (Confidence Interval):
---------- python -m src.evaluate_robustness
-- 4. Kịch bản Bác sĩ can thiệp (Concept Intervention): Đo lường sự gia tăng F1-Score khi có sự hợp tác giữa AI và Bác sĩ:
---------- python -m src.sequential_cbm
