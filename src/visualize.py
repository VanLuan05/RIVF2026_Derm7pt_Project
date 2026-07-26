import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="disease_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    
    # Hiển thị TÊN BỆNH trực tiếp lên biểu đồ
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix - Derm7pt (5 Classes)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label (Thực tế)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label (Dự đoán)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right') # Xoay nghiêng tên bệnh cho dễ đọc
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) 
    plt.close()
    print(f"📸 Đã lưu biểu đồ Ma trận nhầm lẫn tại: {save_path}")

def plot_roc_curve(y_true, y_probs, concept_names, save_path="concept_roc_curve.png"):
    plt.figure(figsize=(10, 8))
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, name in enumerate(concept_names):
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tỷ lệ dương tính giả)', fontsize=12)
    plt.ylabel('True Positive Rate (Tỷ lệ dương tính thật)', fontsize=12)
    plt.title('ROC Curve - Clinical Concepts (Abnormal Detection)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ ROC Curve tại: {save_path}")

def main():
    base_dir = "data/"
    model_path = "best_model.pth"
        
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    LABEL_MAPPING_JSON = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    print("Đang đọc dữ liệu từ JSON...")
    with open(LABEL_MAPPING_JSON, 'r') as f:
        disease_to_idx = json.load(f)
        
    num_disease_classes = len(disease_to_idx)
    disease_names = [k for k, v in sorted(disease_to_idx.items(), key=lambda item: item[1])]
    
    val_dataset = MultimodalDermDataset(
        csv_file=VAL_CSV, img_dir=IMG_DIR, 
        label_mapping_path=LABEL_MAPPING_JSON, transform=test_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = MultimodalDermModel(num_classes=num_disease_classes, num_concepts=7, use_metadata=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Đã nạp trọng số từ: {model_path}")
    else:
        print(f"Không tìm thấy {model_path}!")
        return

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
            
            d_probs = torch.softmax(d_logits, dim=1)
            d_preds = torch.argmax(d_probs, dim=1).cpu().numpy()
            all_disease_labels.extend(d_labels)
            all_disease_preds.extend(d_preds)
            
            c_probs = torch.sigmoid(c_logits).cpu().numpy()
            all_concept_labels.extend(c_labels)
            all_concept_probs.extend(c_probs)

    print("\nBắt đầu vẽ biểu đồ...")
    # Vẽ Confusion Matrix đẹp mắt với Tên Bệnh
    plot_confusion_matrix(all_disease_labels, all_disease_preds, disease_names)
    
    concept_names = [
        'Pigment Network', 'Streaks', 'Pigmentation', 
        'Regression Structures', 'Dots and Globules', 
        'Blue Whitish Veil', 'Vascular Structures'
    ]
    plot_roc_curve(all_concept_labels, all_concept_probs, concept_names)
    print("\nHoàn tất! Các biểu đồ đã được lưu trong thư mục hiện tại.")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()