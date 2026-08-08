import os
import json
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                             precision_score, recall_score, roc_auc_score, confusion_matrix)
from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def plot_normalized_confusion_matrix(cm, classes, save_path, title):
    """Hàm vẽ và lưu Normalized Confusion Matrix"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Tránh lỗi chia cho 0
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def evaluate_single_model(model, data_loader, device, num_classes=5):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            clinic_img, derm_img = batch['clinic_img'].to(device), batch['derm_img'].to(device)
            meta_features = batch['metadata'].to(device)
            labels = batch['label_disease'].numpy()
            
            disease_logits, _ = model(clinic_img, derm_img, meta_features=meta_features)
            probs = torch.softmax(disease_logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 1. Các chỉ số Cơ bản & Nâng cao (Macro Average)
    acc = accuracy_score(all_labels, all_preds)
    b_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auroc = np.nan
    
    # 2. Tính Sensitivity (Recall) và Specificity trên từng lớp
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    sensitivity, specificity = [], []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivity.append(sens)
        specificity.append(spec)

    macro_specificity = np.mean(specificity)
    
    metrics = {
        "acc": acc, "b_acc": b_acc, "f1": f1,
        "precision": precision, "recall": recall, 
        "auroc": auroc, "macro_spec": macro_specificity,
        "cm": cm, "sens_per_class": sensitivity, "spec_per_class": specificity
    }
    return metrics

def main():
    print("="*60)
    print("BẮT ĐẦU CHẤM ĐIỂM (ADVANCED EVALUATION) TRÊN TẬP TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TEST_CSV = os.path.join(base_dir, "processed/test_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/"
    
    # Load Label Mapping
    with open(LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
    disease_names = [k for k, v in sorted(disease_to_idx.items(), key=lambda item: item[1])]
    
    test_dataset = MultimodalDermDataset(TEST_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    workers = 2 if "content" in base_dir else 0 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    try:
        encoder = joblib.load(os.path.join(OUTPUT_DIR, "meta_encoder.joblib"))
        dynamic_meta_dim = len(encoder.get_feature_names_out())
    except:
        dynamic_meta_dim = 14

    experiments = [
        {"name": "B1_Clinical_Only", "modality": "clinic_only", "bottleneck": "none", "meta": False},
        {"name": "B2_Derm_Only",     "modality": "derm_only",   "bottleneck": "none", "meta": False},
        {"name": "B3_Meta_Only",     "modality": "meta_only",   "bottleneck": "none", "meta": True},
        {"name": "B4_Dual_NoMeta",   "modality": "dual",        "bottleneck": "none", "meta": False},
        {"name": "B5_Dual_Metadata", "modality": "dual",        "bottleneck": "none", "meta": True},
        {"name": "B6_PureCBM",       "modality": "dual",        "bottleneck": "pure", "meta": True},
        {"name": "Proposed_Hybrid",  "modality": "dual",        "bottleneck": "hybrid", "meta": True}
    ]
    seeds = [42, 100, 2026]
    
    results_macro = []
    
    for exp in experiments:
        exp_name = exp["name"]
        print(f"\nĐang đánh giá mô hình: {exp_name}")
        
        metrics_dict = {"acc": [], "b_acc": [], "f1": [], "precision": [], "recall": [], "auroc": [], "spec": []}
        best_seed_cm = None 
        best_seed_f1 = -1
        
        for seed in seeds:
            model_path = os.path.join(OUTPUT_DIR, f"{exp_name}_seed_{seed}.pth")
            if not os.path.exists(model_path):
                print(f"  -> Bỏ qua Seed {seed}: Không tìm thấy file")
                continue
                
            model = MultimodalDermModel(
                num_classes=5, num_concepts=7, 
                modality=exp["modality"], bottleneck_type=exp["bottleneck"], 
                use_metadata=exp["meta"], meta_input_dim=dynamic_meta_dim
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            res = evaluate_single_model(model, test_loader, device, num_classes=5)
            metrics_dict["acc"].append(res["acc"])
            metrics_dict["b_acc"].append(res["b_acc"])
            metrics_dict["f1"].append(res["f1"])
            metrics_dict["precision"].append(res["precision"])
            metrics_dict["recall"].append(res["recall"])
            metrics_dict["auroc"].append(res["auroc"])
            metrics_dict["spec"].append(res["macro_spec"])
            
            # Lưu lại CM của seed tốt nhất để vẽ biểu đồ
            if res["f1"] > best_seed_f1:
                best_seed_f1 = res["f1"]
                best_seed_cm = res["cm"]
            
            print(f"  -> Seed {seed} | F1: {res['f1']:.4f} | AUROC: {res['auroc']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f}")
            
        if metrics_dict["acc"]:
            # Lưu tóm tắt Macro vào list
            results_macro.append({
                "Model": exp_name,
                "Accuracy": f"{np.mean(metrics_dict['acc']):.4f} ± {np.std(metrics_dict['acc']):.4f}",
                "Macro F1": f"{np.mean(metrics_dict['f1']):.4f} ± {np.std(metrics_dict['f1']):.4f}",
                "Macro Precision": f"{np.mean(metrics_dict['precision']):.4f} ± {np.std(metrics_dict['precision']):.4f}",
                "Macro Recall": f"{np.mean(metrics_dict['recall']):.4f} ± {np.std(metrics_dict['recall']):.4f}",
                "Macro Specificity": f"{np.mean(metrics_dict['spec']):.4f} ± {np.std(metrics_dict['spec']):.4f}",
                "One-vs-Rest AUROC": f"{np.mean(metrics_dict['auroc']):.4f} ± {np.std(metrics_dict['auroc']):.4f}"
            })
            
            # Lưu ảnh Normalized Confusion Matrix cho mô hình
            if best_seed_cm is not None:
                cm_path = os.path.join(OUTPUT_DIR, f"{exp_name}_best_normalized_cm.png")
                plot_normalized_confusion_matrix(best_seed_cm, disease_names, cm_path, f"Normalized CM - {exp_name}")

    # Xuất báo cáo tổng hợp
    df_results = pd.DataFrame(results_macro)
    csv_out_path = os.path.join(OUTPUT_DIR, "final_advanced_results_summary.csv")
    df_results.to_csv(csv_out_path, index=False)
    
    print("\n" + "="*60)
    print(f"ĐÃ HOÀN TẤT ĐÁNH GIÁ NÂNG CAO!")
    print(f"Bảng kết quả tổng hợp: {csv_out_path}")
    print(f"Biểu đồ Ma trận nhầm lẫn (Normalized) đã được lưu thành file PNG trong {OUTPUT_DIR}")
    print("="*60)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()