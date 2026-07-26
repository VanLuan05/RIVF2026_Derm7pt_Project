import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np

# Nhập các module dự án
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def evaluate_model(model, test_loader, device):
    """
    Hàm đánh giá toàn diện mô hình đa phương thức.
    """
    print(f"Đang tiến hành đánh giá trên thiết bị: {device}...")
    model.to(device)
    model.eval() # Bật chế độ đánh giá (đóng băng Dropout, BatchNorm)
    
    # Các danh sách lưu trữ kết quả thực tế và dự đoán
    all_disease_preds = []
    all_disease_labels = []
    
    all_concept_preds = []
    all_concept_labels = []
    all_concept_probs = [] # Lưu xác suất thô để tính ROC-AUC

    with torch.no_grad(): # Tắt tính toán đạo hàm để tiết kiệm RAM
        for batch in test_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            
            disease_labels = batch['label_disease'].cpu().numpy()
            concept_labels = batch['concept_labels'].cpu().numpy()
            
            # Chạy suy luận (Inference)
            disease_logits, concept_logits = model(clinic_img, derm_img, meta_features=None)
            
            # 1. Xử lý dự đoán Bệnh (Bài toán Multiclass - Phân loại nhiều lớp)
            disease_probs = torch.softmax(disease_logits, dim=1)
            disease_preds = torch.argmax(disease_probs, dim=1).cpu().numpy()
            
            all_disease_labels.extend(disease_labels)
            all_disease_preds.extend(disease_preds)
            
            # 2. Xử lý dự đoán Khái niệm (Bài toán Multilabel - Phân loại đa nhãn)
            concept_probs = torch.sigmoid(concept_logits)
            
            # HẠ NGƯỠNG XUỐNG 0.3 (30% tin tưởng là kết luận có triệu chứng)
            concept_preds = (concept_probs > 0.3).int().cpu().numpy()
            
            all_concept_labels.extend(concept_labels)
            all_concept_preds.extend(concept_preds)
            all_concept_probs.extend(concept_probs.cpu().numpy())

    # ==========================================
    # TÍNH TOÁN CHỈ SỐ BẰNG SCIKIT-LEARN
    # ==========================================
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CHẨN ĐOÁN BỆNH")
    print("="*50)
    
    # Tính Accuracy và F1-Score (Macro) cho bệnh
    acc_disease = accuracy_score(all_disease_labels, all_disease_preds)
    f1_disease = f1_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    
    print(f"Accuracy (Độ chính xác): {acc_disease:.4f}")
    print(f"F1-Score (Macro):        {f1_disease:.4f}")
    print("\nChi tiết từng lớp bệnh (Classification Report):")
    print(classification_report(all_disease_labels, all_disease_preds, zero_division=0))


    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ KHÁI NIỆM LÂM SÀNG (EXPLAINABILITY)")
    print("="*50)
    
    # Tính ROC-AUC (Chỉ số đánh giá độ tin cậy của xác suất)
    try:
        auc_score = roc_auc_score(all_concept_labels, all_concept_probs, average='macro')
        print(f"ROC-AUC Khái niệm (Càng gần 1 càng tốt): {auc_score:.4f}")
    except ValueError:
        print("ROC-AUC: Không thể tính do có lớp thiểu số thiếu mẫu ở tập Val.")

    # Tính F1-Score (Macro) cho 7 khái niệm với ngưỡng mới
    f1_concepts = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
    print(f"F1-Score Khái niệm (Ngưỡng 0.3): {f1_concepts:.4f}")
    print("\nChi tiết 7 Khái niệm (Classification Report):")
    
    concept_names = [
        'pigment_network', 'streaks', 'pigmentation', 
        'regression_structures', 'dots_and_globules', 
        'blue_whitish_veil', 'vascular_structures'
    ]
    print(classification_report(all_concept_labels, all_concept_preds, target_names=concept_names, zero_division=0))


def main():
    # 1. Tự động phát hiện môi trường 
    colab_drive_path = "/content/drive/MyDrive/RIVF2026_Dataset/data/"
    if os.path.exists(colab_drive_path):
        base_dir = colab_drive_path
        model_path = "/content/drive/MyDrive/RIVF2026_Dataset/best_model.pth" 
    else:
        base_dir = "data/"
        model_path = "best_model.pth"
        
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    # 2. Đồng bộ từ điển bệnh từ tập Train sang tập Validation
    print("Đang đồng bộ từ điển nhãn bệnh...")
    train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=None)
    val_dataset = MultimodalDermDataset(csv_file=VAL_CSV, img_dir=IMG_DIR, transform=test_transforms)
    
    # Ép tập Validation phải dùng chung hệ thống đánh số của tập Train
    val_dataset.disease_to_idx = train_dataset.disease_to_idx
    num_disease_classes = len(train_dataset.disease_to_idx)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Khởi tạo lại "cái xác" của mô hình
    model = MultimodalDermModel(num_classes=num_disease_classes, num_concepts=7, use_metadata=False)
    
    # 4. Bơm "linh hồn" (Trọng số đã huấn luyện) vào
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã tải thành công trọng số từ: {model_path}")
    else:
        print(f"Không tìm thấy file trọng số tại {model_path}. Vui lòng kiểm tra lại!")
        return

    # 5. Kích hoạt đánh giá
    evaluate_model(model, val_loader, device)


if __name__ == "__main__":
    main()