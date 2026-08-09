import os
import json
import torch
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tự động nhận diện đường dẫn dữ liệu (Colab hoặc Local)
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    TEST_CSV = os.path.join(base_dir, "processed/test_split.csv") # Dùng tập Test theo chuẩn đánh giá cuối
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/"

    # 2. Đọc số chiều Metadata tự động (KHẮC PHỤC LỖI P0)
    try:
        encoder = joblib.load(os.path.join(OUTPUT_DIR, "meta_encoder.joblib"))
        dynamic_meta_dim = len(encoder.get_feature_names_out())
        print(f"[*] Số chiều Metadata tự động nhận diện: {dynamic_meta_dim}")
    except:
        print("[!] Không tìm thấy meta_encoder.joblib, dùng mặc định 14 chiều.")
        dynamic_meta_dim = 14

    # 3. Tải mô hình Pure (ĐÃ ĐỔI TÊN THÀNH B6_PureCBM THEO CHUẨN MỚI)
    seed = 42
    model_name = f"B6_PureCBM_seed_{seed}"
    model_path = os.path.join(OUTPUT_DIR, f"{model_name}.pth")
    
    with open(LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
        
    model = MultimodalDermModel(
        num_classes=len(disease_to_idx), num_concepts=7, 
        modality='dual', bottleneck_type='pure', use_metadata=True, meta_input_dim=dynamic_meta_dim
    )
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file trọng số {model_path}! Hãy chắc chắn bạn đã chạy train ablation xong.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Load Dữ liệu
    print("Đang quét dữ liệu... Vui lòng đợi khoảng 1-2 phút...")
    workers = 2 if "content" in base_dir else 0
    train_dataset = MultimodalDermDataset(TRAIN_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    test_dataset = MultimodalDermDataset(TEST_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=workers)

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

    y_train_d, y_train_c, _, X_train_meta = extract_all(train_loader)
    y_test_d, y_test_c, X_test_pred_c, X_test_meta = extract_all(test_loader)

    # 5. Huấn luyện Bác sĩ Độc lập (Sequential Classifier)
    print("\nĐang huấn luyện Mô hình Chuỗi (Sequential CBM)...")
    X_train_perfect = np.hstack((y_train_c, X_train_meta))
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_perfect, y_train_d)

    # 6. Đánh giá TRƯỚC Can Thiệp (Dùng Concept lờ mờ do AI đoán)
    X_test_ai = np.hstack((X_test_pred_c, X_test_meta))
    preds_before = clf.predict(X_test_ai)
    acc_before = accuracy_score(y_test_d, preds_before)
    f1_before = f1_score(y_test_d, preds_before, average='macro')

    # 7. Đánh giá SAU Can Thiệp (Dùng Concept chuẩn của Bác sĩ)
    X_test_doctor = np.hstack((y_test_c, X_test_meta))
    preds_after = clf.predict(X_test_doctor)
    acc_after = accuracy_score(y_test_d, preds_after)
    f1_after = f1_score(y_test_d, preds_after, average='macro')

    # XUẤT KẾT QUẢ VÀ LƯU FILE MARKDOWN (Khắc phục việc CSV bị ignore trên GitHub)
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

    # Lưu Markdown để commit lên GitHub
    results_df = pd.DataFrame([{
        "Model": "Sequential CBM (Based on B6_PureCBM)",
        "Metric": "Macro F1",
        "Before Intervention (AI)": f"{f1_before:.4f}",
        "After Intervention (Doctor)": f"{f1_after:.4f}",
        "Improvement": f"+{(f1_after - f1_before):.4f}"
    }])
    
    os.makedirs("results", exist_ok=True)
    md_path = "results/sequential_cbm_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("### Sequential CBM Intervention Results\n\n")
        f.write(results_df.to_markdown(index=False))
        
    print(f"Đã lưu báo cáo Markdown tại: {md_path}")

if __name__ == "__main__":
    main()