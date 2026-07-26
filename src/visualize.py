import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from torch.utils.data import DataLoader

# Nhập các module dự án
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def plot_confusion_matrix(y_true, y_pred, num_classes, save_path="disease_confusion_matrix.png"):
    """Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)"""
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(12, 10))
    # Sử dụng heatmap của seaborn để biểu đồ trông chuyên nghiệp hơn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    
    plt.title('Confusion Matrix - Disease Diagnosis', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label (Thực tế)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label (Dự đoán)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # Lưu với độ phân giải cao 300 DPI để in báo cáo
    plt.close()
    print(f"Đã lưu biểu đồ Ma trận nhầm lẫn tại: {save_path}")

def plot_roc_curve(y_true, y_probs, concept_names, save_path="concept_roc_curve.png"):
    """Vẽ và lưu Đường cong ROC cho 7 Khái niệm lâm sàng"""
    plt.figure(figsize=(10, 8))
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, name in enumerate(concept_names):
        # Chỉ vẽ đường ROC nếu lớp đó có cả mẫu âm tính và dương tính
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
    # Vẽ đường chéo tham chiếu (mức độ đoán mò ngẫu nhiên)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tỷ lệ dương tính giả)', fontsize=12)
    plt.ylabel('True Positive Rate (Tỷ lệ dương tính thật)', fontsize=12)
    plt.title('ROC Curve - Clinical Concepts', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ ROC Curve tại: {save_path}")

def main():
    # 1. Cấu hình đường dẫn (chạy cục bộ trên máy tính)
    base_dir = "data/"
    model_path = "best_model.pth"
        
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    # 2. Nạp dữ liệu
    print("Đang nạp dữ liệu để trích xuất biểu đồ...")
    train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=None)
    val_dataset = MultimodalDermDataset(csv_file=VAL_CSV, img_dir=IMG_DIR, transform=test_transforms)
    val_dataset.disease_to_idx = train_dataset.disease_to_idx
    num_disease_classes = len(train_dataset.disease_to_idx)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Khởi tạo mô hình
    model = MultimodalDermModel(num_classes=num_disease_classes, num_concepts=7, use_metadata=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã nạp trọng số từ: {model_path}")
    else:
        print(f"Không tìm thấy {model_path}!")
        return

    # 4. Chạy mô hình để lấy dự đoán
    print("Đang phân tích các ca bệnh, vui lòng đợi...")
    model.to(device)
    model.eval()
    
    all_disease_labels, all_disease_preds = [], []
    all_concept_labels, all_concept_probs = [], []

    with torch.no_grad():
        for batch in val_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            
            d_labels = batch['label_disease'].cpu().numpy()
            c_labels = batch['concept_labels'].cpu().numpy()
            
            d_logits, c_logits = model(clinic_img, derm_img, meta_features=None)
            
            # Bệnh
            d_probs = torch.softmax(d_logits, dim=1)
            d_preds = torch.argmax(d_probs, dim=1).cpu().numpy()
            all_disease_labels.extend(d_labels)
            all_disease_preds.extend(d_preds)
            
            # Khái niệm
            c_probs = torch.sigmoid(c_logits).cpu().numpy()
            all_concept_labels.extend(c_labels)
            all_concept_probs.extend(c_probs)

    # 5. Khởi chạy vẽ biểu đồ
    print("\nBắt đầu vẽ biểu đồ...")
    plot_confusion_matrix(all_disease_labels, all_disease_preds, num_disease_classes)
    
    concept_names = [
        'Pigment Network', 'Streaks', 'Pigmentation', 
        'Regression Structures', 'Dots and Globules', 
        'Blue Whitish Veil', 'Vascular Structures'
    ]
    plot_roc_curve(all_concept_labels, all_concept_probs, concept_names)
    
    print("\nHoàn tất! Hãy kiểm tra thư mục gốc dự án của bạn để lấy ảnh báo cáo.")

if __name__ == "__main__":
    # Import các thư viện cần thiết để tránh lỗi font chữ khi vẽ
    import matplotlib
    matplotlib.use('Agg')
    main()