import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def main():
    # CHÚ Ý: Phải chọn một mô hình cấu trúc Hybrid (Ví dụ bạn lấy Baseline B4 hoặc B5)
    # Giả sử bạn có file 'best_model_hybrid.pth' trong thư mục outputs
    model_name = "best_model_pure" 
    model_path = Config.get_checkpoint_path(experiment_name=model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(Config.LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
        
    val_dataset = MultimodalDermDataset(
        csv_file=Config.VAL_CSV, img_dir=Config.IMG_DIR, 
        label_mapping_path=Config.LABEL_MAPPING, transform=test_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Khởi tạo mô hình Hybrid
    model = MultimodalDermModel(
        num_classes=len(disease_to_idx), 
        num_concepts=7, 
        modality='dual',
        bottleneck_type='pure', # <--- Bắt buộc phải là hybrid hoặc pure
        use_metadata=True
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Đã nạp trọng số Hybrid từ: {model_path}")
    else:
        print(f"Không tìm thấy {model_path}! Bạn cần train một mô hình Hybrid trước.")
        return

    model.to(device)
    model.eval()

    all_labels = []
    preds_before = []
    preds_after = []

    print("\nĐang chạy Giả lập Can thiệp Khái niệm (Concept Intervention)...")
    
    with torch.no_grad():
        for batch in val_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            meta_features = batch['metadata'].to(device)
            
            d_labels = batch['label_disease'].to(device)
            c_labels = batch['concept_labels'].to(device).float() # Nhãn concept chuẩn của bác sĩ
            
            # LẦN 1: AI tự đoán hoàn toàn (Không can thiệp)
            logits_before, _ = model(clinic_img, derm_img, meta_features=meta_features)
            probs_before = torch.softmax(logits_before, dim=1)
            preds_before.extend(torch.argmax(probs_before, dim=1).cpu().numpy())
            
            # LẦN 2: Bác sĩ can thiệp (Nhét c_labels chuẩn vào mô hình)
            logits_after, _ = model(clinic_img, derm_img, meta_features=meta_features, intervention_probs=c_labels)
            probs_after = torch.softmax(logits_after, dim=1)
            preds_after.extend(torch.argmax(probs_after, dim=1).cpu().numpy())
            
            all_labels.extend(d_labels.cpu().numpy())

    # --- TÍNH TOÁN VÀ IN BÁO CÁO ---
    acc_before = accuracy_score(all_labels, preds_before)
    f1_before = f1_score(all_labels, preds_before, average='macro')
    
    acc_after = accuracy_score(all_labels, preds_after)
    f1_after = f1_score(all_labels, preds_after, average='macro')

    print("="*60)
    print(" KẾT QUẢ KỊCH BẢN BÁC SĨ CAN THIỆP (INTERVENTION)")
    print("="*60)
    print(f"[TRƯỚC CAN THIỆP] - AI tự quyết định:")
    print(f" - Accuracy : {acc_before:.4f}")
    print(f" - F1-Score : {f1_before:.4f}")
    print("-" * 60)
    print(f"[SAU CAN THIỆP] - Bác sĩ sửa đúng 100% Concept:")
    print(f" - Accuracy : {acc_after:.4f}  (Tăng +{(acc_after - acc_before):.4f})")
    print(f" - F1-Score : {f1_after:.4f}  (Tăng +{(f1_after - f1_before):.4f})")
    print("="*60)
    print("Kết luận: Bác sĩ và AI hợp tác sẽ mang lại kết quả chẩn đoán cao nhất!")

if __name__ == "__main__":
    main()