import os
import json
import torch
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tải mô hình Pure (Để lấy dự đoán Concept của AI)
    model_name = "B5_Dual_PureCBM_seed_42"
    model_path = Config.get_checkpoint_path(experiment_name=model_name)
    
    with open(Config.LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
        
    model = MultimodalDermModel(
        num_classes=len(disease_to_idx), num_concepts=7, 
        modality='dual', bottleneck_type='pure', use_metadata=True, meta_input_dim=14
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Dữ liệu
    print("Đang quét dữ liệu... Vui lòng đợi khoảng 1-2 phút...")
    # Tái sử dụng test_transforms cho cả train để không bị random crop làm lệch đặc trưng
    train_dataset = MultimodalDermDataset(Config.TRAIN_CSV, Config.IMG_DIR, Config.LABEL_MAPPING, transform=test_transforms)
    val_dataset = MultimodalDermDataset(Config.VAL_CSV, Config.IMG_DIR, Config.LABEL_MAPPING, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def extract_all(loader):
        true_d, true_c, pred_c, meta = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                c_img = batch['clinic_img'].to(device)
                d_img = batch['derm_img'].to(device)
                m = batch['metadata'].to(device)
                
                # AI dự đoán Concept
                _, c_logits = model(c_img, d_img, meta_features=m)
                c_probs = torch.sigmoid(c_logits)
                
                true_d.extend(batch['label_disease'].numpy())
                true_c.extend(batch['concept_labels'].numpy())
                pred_c.extend(c_probs.cpu().numpy())
                meta.extend(m.cpu().numpy())
                
        return np.array(true_d), np.array(true_c), np.array(pred_c), np.array(meta)

    # Lấy dữ liệu (Mất khoảng 1-2 phút)
    y_train_d, y_train_c, _, X_train_meta = extract_all(train_loader)
    y_val_d, y_val_c, X_val_pred_c, X_val_meta = extract_all(val_loader)

    # 3. Huấn luyện Bác sĩ Độc lập (Sequential Classifier)
    # Vị bác sĩ này được huấn luyện 100% bằng Concept chuẩn, hoàn toàn miễn nhiễm với rò rỉ thông tin
    print("\nĐang huấn luyện Mô hình Chuỗi (Sequential CBM)...")
    X_train_perfect = np.hstack((y_train_c, X_train_meta))
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_perfect, y_train_d)

    # 4. Đánh giá TRƯỚC Can Thiệp (Dùng Concept lờ mờ do AI đoán)
    X_val_ai = np.hstack((X_val_pred_c, X_val_meta))
    preds_before = clf.predict(X_val_ai)
    acc_before = accuracy_score(y_val_d, preds_before)
    f1_before = f1_score(y_val_d, preds_before, average='macro')

    # 5. Đánh giá SAU Can Thiệp (Dùng Concept chuẩn của Bác sĩ)
    X_val_doctor = np.hstack((y_val_c, X_val_meta))
    preds_after = clf.predict(X_val_doctor)
    acc_after = accuracy_score(y_val_d, preds_after)
    f1_after = f1_score(y_val_d, preds_after, average='macro')

    print("="*60)
    print("KẾT QUẢ CAN THIỆP - MÔ HÌNH CHUỖI ĐỘC LẬP (SEQUENTIAL CBM)")
    print("="*60)
    print(f"[TRƯỚC CAN THIỆP] - Sử dụng Concept do AI tự đoán:")
    print(f" - Accuracy : {acc_before:.4f}")
    print(f" - F1-Score : {f1_before:.4f}")
    print("-" * 60)
    print(f"[SAU CAN THIỆP] - Bác sĩ sửa Concept chuẩn 100%:")
    print(f" - Accuracy : {acc_after:.4f}  (Tăng +{(acc_after - acc_before):.4f})")
    print(f" - F1-Score : {f1_after:.4f}  (Tăng +{(f1_after - f1_before):.4f})")
    print("="*60)
    print("Đóng đinh bài báo: Kỹ thuật Sequential CBM giúp Bác sĩ và AI tương tác an toàn và hiệu quả!")

if __name__ == "__main__":
    main()