import torch
import os
import json
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np

from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel
from src.config import Config

def evaluate_model(model, test_loader, device, disease_names):
    print(f"Đang tiến hành đánh giá trên thiết bị: {device}...")
    model.to(device)
    model.eval()
    
    all_disease_preds, all_disease_labels = [], []
    all_concept_preds, all_concept_labels, all_concept_probs = [], [], []

    with torch.no_grad(): 
        for batch in test_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            
            disease_labels = batch['label_disease'].cpu().numpy()
            concept_labels = batch['concept_labels'].cpu().numpy()
            # --- 1. LẤY METADATA TỪ BATCH ---
            meta_features = batch['metadata'].to(device)
            # Mô hình trả về 2 output
            disease_logits, concept_logits = model(clinic_img, derm_img, meta_features=meta_features)
            
            # 1. Đánh giá Bệnh (Luôn luôn có)
            disease_probs = torch.softmax(disease_logits, dim=1)
            disease_preds = torch.argmax(disease_probs, dim=1).cpu().numpy()
            all_disease_labels.extend(disease_labels)
            all_disease_preds.extend(disease_preds)
            
            # 2. Đánh giá Concept (Chỉ tính nếu mô hình CÓ concept, tức B3, B4, B5)
            if concept_logits is not None:
                concept_probs = torch.sigmoid(concept_logits)
                concept_preds = (concept_probs > 0.5).int().cpu().numpy()
                all_concept_labels.extend(concept_labels)
                all_concept_preds.extend(concept_preds)
                all_concept_probs.extend(concept_probs.cpu().numpy())

    # --- IN KẾT QUẢ BỆNH ---
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ CHẨN ĐOÁN BỆNH (5 LỚP CHUẨN)")
    print("="*50)
    acc_disease = accuracy_score(all_disease_labels, all_disease_preds)
    f1_disease = f1_score(all_disease_labels, all_disease_preds, average='macro', zero_division=0)
    
    print(f"Accuracy (Độ chính xác): {acc_disease:.4f}")
    print(f"F1-Score (Macro):        {f1_disease:.4f}")
    print("\nChi tiết từng lớp bệnh (Classification Report):")
    print(classification_report(all_disease_labels, all_disease_preds, target_names=disease_names, zero_division=0))

    # --- IN KẾT QUẢ CONCEPT (Nếu có) ---
    if len(all_concept_preds) > 0:
        print("\n" + "="*50)
        print("KẾT QUẢ ĐÁNH GIÁ KHÁI NIỆM LÂM SÀNG (EXPLAINABILITY)")
        print("="*50)
        try:
            auc_score = roc_auc_score(all_concept_labels, all_concept_probs, average='macro')
            print(f"ROC-AUC Khái niệm (Càng gần 1 càng tốt): {auc_score:.4f}")
        except ValueError:
            print("ROC-AUC: Không thể tính do thiếu mẫu.")
        f1_concepts = f1_score(all_concept_labels, all_concept_preds, average='macro', zero_division=0)
        print(f"F1-Score Khái niệm (Ngưỡng 0.5): {f1_concepts:.4f}")
        
        concept_names = ['Pigment Network', 'Streaks', 'Pigmentation', 'Regression Structures', 'Dots and Globules', 'Blue Whitish Veil', 'Vascular Structures']
        print(classification_report(all_concept_labels, all_concept_preds, target_names=concept_names, zero_division=0))
    else:
        print("\n[Mô hình Baseline này không dự đoán Khái niệm Lâm sàng]")

def main():
    # Lấy đường dẫn lưu trọng số chuẩn xác qua Config
    model_name = "baseline_meta_only" 
    model_path = Config.get_checkpoint_path(experiment_name=model_name)
        
    VAL_CSV = Config.VAL_CSV
    LABEL_MAPPING_JSON = Config.LABEL_MAPPING
    IMG_DIR = Config.IMG_DIR
    
    with open(LABEL_MAPPING_JSON, 'r') as f:
        disease_to_idx = json.load(f)
        
    num_disease_classes = len(disease_to_idx)
    disease_names = [k for k, v in sorted(disease_to_idx.items(), key=lambda item: item[1])]
    
    val_dataset = MultimodalDermDataset(
        csv_file=VAL_CSV, 
        img_dir=IMG_DIR, 
        label_mapping_path=LABEL_MAPPING_JSON, 
        transform=test_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # =========================================================================
    # LƯU Ý QUAN TRỌNG: Cấu hình dưới đây dành cho Master Model P2
    # =========================================================================
    model = MultimodalDermModel(
        num_classes=num_disease_classes, 
        num_concepts=7, 
        modality='meta_only',
        bottleneck_type='none',
        use_metadata=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Đã tải thành công trọng số từ: {model_path}")
        evaluate_model(model, val_loader, device, disease_names)
    else:
        print(f"Không tìm thấy file trọng số tại {model_path}. Vui lòng chờ huấn luyện xong!")
if __name__ == "__main__":
    main()