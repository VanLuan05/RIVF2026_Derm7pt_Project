import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
import random
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms
from src.models.models import MultimodalDermModel

def set_seed(seed):
    """Khóa toàn bộ tính ngẫu nhiên để đảm bảo khả năng tái lập (Reproducibility)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_eval(alpha, seed, train_loader, val_loader, device, dynamic_meta_dim):
    set_seed(seed)
    print(f"  -> Đang chạy Seed: {seed}...")
    
    model = MultimodalDermModel(
        num_classes=5, num_concepts=7, 
        modality='dual', bottleneck_type='hybrid', 
        use_metadata=True, meta_input_dim=dynamic_meta_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_disease = nn.CrossEntropyLoss()
    criterion_concept = nn.BCEWithLogitsLoss()
    
    epochs = 15
    
    # Huấn luyện
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            c_img, d_img = batch['clinic_img'].to(device), batch['derm_img'].to(device)
            meta = batch['metadata'].to(device)
            labels_d = batch['label_disease'].to(device)
            labels_c = batch['concept_labels'].to(device).float()
            
            optimizer.zero_grad()
            logits_d, logits_c = model(c_img, d_img, meta_features=meta)
            
            loss = criterion_disease(logits_d, labels_d) + alpha * criterion_concept(logits_c, labels_c)
            loss.backward()
            optimizer.step()
            
    # Đánh giá trên Validation
    model.eval()
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            c_img, d_img = batch['clinic_img'].to(device), batch['derm_img'].to(device)
            meta = batch['metadata'].to(device)
            
            logits_d, logits_c = model(c_img, d_img, meta_features=meta)
            
            preds_d = torch.argmax(torch.softmax(logits_d, dim=1), dim=1).cpu().numpy()
            all_preds_d.extend(preds_d)
            all_labels_d.extend(batch['label_disease'].numpy())
            
            preds_c = (torch.sigmoid(logits_c) > 0.5).int().cpu().numpy()
            all_preds_c.extend(preds_c)
            all_labels_c.extend(batch['concept_labels'].numpy())
            
    f1_disease = f1_score(all_labels_d, all_preds_d, average='macro', zero_division=0)
    f1_concept = f1_score(all_labels_c, all_preds_c, average='macro', zero_division=0)
    
    print(f"     [+] Hoàn thành Seed {seed} | Disease F1: {f1_disease:.4f} | Concept F1: {f1_concept:.4f}")
    return f1_disease, f1_concept

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    IMG_DIR = "/content/local_images/" if os.path.exists("/content/local_images/") else os.path.join(base_dir, "raw/images/")
    
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    
    try:
        encoder = joblib.load("outputs/meta_encoder.joblib")
        dynamic_meta_dim = len(encoder.get_feature_names_out())
    except:
        dynamic_meta_dim = 14
        
    train_dataset = MultimodalDermDataset(TRAIN_CSV, IMG_DIR, LABEL_MAPPING, transform=train_transforms)
    val_dataset = MultimodalDermDataset(VAL_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    
    workers = 2 if "content" in base_dir else 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    alphas = [2.0, 3.0]
    seeds = [42, 100, 2026]
    final_results = []
    
    print("="*65)
    print("CHUNG KẾT ALPHA ABLATION: Alpha = 2.0 vs Alpha = 3.0 (3 Seeds)")
    print("="*65)
    
    for alpha in alphas:
        print(f"\n[>>>] ĐANG ĐÁNH GIÁ ALPHA = {alpha}")
        d_f1_list, c_f1_list = [], []
        
        for seed in seeds:
            d_f1, c_f1 = train_and_eval(alpha, seed, train_loader, val_loader, device, dynamic_meta_dim)
            d_f1_list.append(d_f1)
            c_f1_list.append(c_f1)
            
        final_results.append({
            "Alpha": alpha,
            "Seed 42 (Disease F1)": f"{d_f1_list[0]:.4f}",
            "Seed 100 (Disease F1)": f"{d_f1_list[1]:.4f}",
            "Seed 2026 (Disease F1)": f"{d_f1_list[2]:.4f}",
            "Mean ± SD (Disease F1)": f"{np.mean(d_f1_list):.4f} ± {np.std(d_f1_list):.4f}",
            "Mean Concept F1": f"{np.mean(c_f1_list):.4f}"
        })
        
    df_results = pd.DataFrame(final_results)
    print("\n" + "="*80)
    print("BẢNG KẾT QUẢ CHUNG CUỘC (QUYẾT ĐỊNH SIÊU THAM SỐ)")
    print("="*80)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()