import os
import json
import torch
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def evaluate_single_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            meta_features = batch['metadata'].to(device)
            labels = batch['label_disease'].numpy()
            
            # Forward pass
            disease_logits, _ = model(clinic_img, derm_img, meta_features=meta_features)
            preds = torch.argmax(torch.softmax(disease_logits, dim=1), dim=1).cpu().numpy()
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            
    acc = accuracy_score(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return acc, b_acc, f1

def main():
    print("="*60)
    print("BẮT ĐẦU CHẤM ĐIỂM (EVALUATION) TRÊN TẬP TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đường dẫn cấu hình
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TEST_CSV = os.path.join(base_dir, "processed/test_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/"
    
    # 2. Load Dữ liệu Test
    test_dataset = MultimodalDermDataset(TEST_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    workers = 2 if "content" in base_dir else 0 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    # 3. Kịch bản các mô hình cần đánh giá
    experiments = [
        {"name": "B1_Clinical_Only", "modality": "clinic_only", "bottleneck": "none", "meta": False},
        {"name": "B2_Derm_Only",     "modality": "derm_only",   "bottleneck": "none", "meta": False},
        {"name": "B3_Meta_Only",     "modality": "meta_only",   "bottleneck": "none", "meta": True},
        {"name": "B4_Dual_NoMeta",   "modality": "dual",        "bottleneck": "none", "meta": False},
        {"name": "B5_Dual_PureCBM",  "modality": "dual",        "bottleneck": "pure", "meta": True},
        {"name": "Master_P2",        "modality": "dual",        "bottleneck": "multitask", "meta": True}
    ]
    seeds = [42, 100, 2026]
    
    results = []

    # 4. Quá trình chấm điểm
    for exp in experiments:
        exp_name = exp["name"]
        print(f"\nĐang đánh giá mô hình: {exp_name}")
        
        acc_list, b_acc_list, f1_list = [], [], []
        
        for seed in seeds:
            model_path = os.path.join(OUTPUT_DIR, f"{exp_name}_seed_{seed}.pth")
            
            if not os.path.exists(model_path):
                print(f"  -> Bỏ qua Seed {seed}: Không tìm thấy file {model_path}")
                continue
                
            # Khởi tạo kiến trúc và tải trọng số (Strict mặc định là True)
            model = MultimodalDermModel(
                num_classes=5, num_concepts=7, 
                modality=exp["modality"], bottleneck_type=exp["bottleneck"], use_metadata=exp["meta"]
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            acc, b_acc, f1 = evaluate_single_model(model, test_loader, device)
            acc_list.append(acc)
            b_acc_list.append(b_acc)
            f1_list.append(f1)
            
            print(f"  -> Seed {seed} | Acc: {acc:.4f} | B-Acc: {b_acc:.4f} | F1: {f1:.4f}")
            
        # Tính Mean ± SD
        if acc_list:
            results.append({
                "Model": exp_name,
                "Accuracy (Mean ± SD)": f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}",
                "Balanced Acc (Mean ± SD)": f"{np.mean(b_acc_list):.4f} ± {np.std(b_acc_list):.4f}",
                "Macro F1 (Mean ± SD)": f"{np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}"
            })

    # 5. Lưu báo cáo ra file CSV
    df_results = pd.DataFrame(results)
    csv_out_path = os.path.join(OUTPUT_DIR, "final_results_summary.csv")
    df_results.to_csv(csv_out_path, index=False)
    
    print("\n" + "="*60)
    print(f"ĐÃ HOÀN TẤT! Bảng kết quả tổng hợp được lưu tại: {csv_out_path}")
    print("="*60)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()