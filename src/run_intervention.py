import os
import json
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def evaluate_intervention(model, data_loader, device):
    model.eval()
    all_labels = []
    preds_ai_only = []
    preds_with_doctor = []
    
    with torch.no_grad():
        for batch in data_loader:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            meta_features = batch['metadata'].to(device)
            labels = batch['label_disease'].numpy()
            
            # Nhãn Concept "vàng" (Ground truth) được coi như Bác sĩ khám chuẩn 100%
            doctor_concepts = batch['concept_labels'].to(device) 
            
            # KỊCH BẢN 1: AI tự chẩn đoán toàn bộ
            logits_ai, _ = model(clinic_img, derm_img, meta_features=meta_features)
            p_ai = torch.argmax(torch.softmax(logits_ai, dim=1), dim=1).cpu().numpy()
            
            # KỊCH BẢN 2: Bác sĩ can thiệp (Ép mô hình dùng nhãn Concept chuẩn)
            logits_doc, _ = model(clinic_img, derm_img, meta_features=meta_features, intervention_probs=doctor_concepts)
            p_doc = torch.argmax(torch.softmax(logits_doc, dim=1), dim=1).cpu().numpy()
            
            all_labels.extend(labels)
            preds_ai_only.extend(p_ai)
            preds_with_doctor.extend(p_doc)
            
    f1_ai = f1_score(all_labels, preds_ai_only, average='macro')
    f1_doc = f1_score(all_labels, preds_with_doctor, average='macro')
    
    return f1_ai, f1_doc

def main():
    print("="*60)
    print("MÔ PHỎNG BÁC SĨ CAN THIỆP LÂM SÀNG (CONCEPT INTERVENTION)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tự động nhận diện đường dẫn dữ liệu (Colab Drive hoặc Local)
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TEST_CSV = os.path.join(base_dir, "processed/test_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/"
    
    test_dataset = MultimodalDermDataset(TEST_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    workers = 2 if "content" in base_dir else 0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    # 2. Đọc số chiều Metadata tự động
    try:
        encoder = joblib.load(os.path.join(OUTPUT_DIR, "meta_encoder.joblib"))
        dynamic_meta_dim = len(encoder.get_feature_names_out())
    except:
        dynamic_meta_dim = 14
    
    # 3. Chỉ test trên B6_PureCBM và Proposed_Hybrid (Khắc phục lỗi P0)
    experiments = [
        {"name": "B6_PureCBM",       "modality": "dual", "bottleneck": "pure",   "meta": True},
        {"name": "Proposed_Hybrid",  "modality": "dual", "bottleneck": "hybrid", "meta": True}
    ]
    seeds = [42, 100, 2026]
    results = []

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\nThử nghiệm can thiệp trên mô hình: {exp_name}")
        
        f1_ai_list, f1_doc_list = [], []
        
        for seed in seeds:
            model_path = os.path.join(OUTPUT_DIR, f"{exp_name}_seed_{seed}.pth")
            if not os.path.exists(model_path):
                print(f"  -> Không tìm thấy file {model_path}")
                continue
                
            model = MultimodalDermModel(
                num_classes=5, num_concepts=7, 
                modality=exp["modality"], bottleneck_type=exp["bottleneck"], 
                use_metadata=exp["meta"], meta_input_dim=dynamic_meta_dim # Đã bổ sung
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            f1_ai, f1_doc = evaluate_intervention(model, test_loader, device)
            f1_ai_list.append(f1_ai)
            f1_doc_list.append(f1_doc)
            
            print(f"  -> Seed {seed} | F1 AI Tự đoán: {f1_ai:.4f} => Có Bác sĩ: {f1_doc:.4f} ({(f1_doc - f1_ai):+.4f})")
            
        if f1_ai_list:
            results.append({
                "Model": exp_name,
                "Macro F1 (AI Only)": f"{np.mean(f1_ai_list):.4f} ± {np.std(f1_ai_list):.4f}",
                "Macro F1 (With Doctor)": f"{np.mean(f1_doc_list):.4f} ± {np.std(f1_doc_list):.4f}",
                "Absolute Improvement": f"+{(np.mean(f1_doc_list) - np.mean(f1_ai_list)):.4f}"
            })

    df_results = pd.DataFrame(results)
    csv_out_path = os.path.join(OUTPUT_DIR, "intervention_results.csv")
    df_results.to_csv(csv_out_path, index=False)
    
    print("\n" + "="*60)
    print("BẢNG BÁO CÁO CONCEPT INTERVENTION (DÁN VÀO BÀI BÁO)")
    print("="*60)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()