import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms
from src.models.models import MultimodalDermModel

def train_and_eval_alpha(alpha, train_loader, val_loader, device, dynamic_meta_dim):
    print(f"\n{'-'*40}")
    print(f"ĐANG HUẤN LUYỆN PROPOSED_HYBRID VỚI ALPHA = {alpha}")
    print(f"{'-'*40}")
    
    # 1. Khởi tạo mô hình mới tinh
    model = MultimodalDermModel(
        num_classes=5, num_concepts=7, 
        modality='dual', bottleneck_type='hybrid', 
        use_metadata=True, meta_input_dim=dynamic_meta_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_disease = nn.CrossEntropyLoss()
    criterion_concept = nn.BCEWithLogitsLoss()
    
    epochs = 15 # Chỉ cần 15 epochs để test lướt xem alpha nào tốt nhất
    
    # 2. Vòng lặp huấn luyện
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            c_img = batch['clinic_img'].to(device)
            d_img = batch['derm_img'].to(device)
            meta = batch['metadata'].to(device)
            labels_d = batch['label_disease'].to(device)
            labels_c = batch['concept_labels'].to(device).float()
            
            optimizer.zero_grad()
            logits_d, logits_c = model(c_img, d_img, meta_features=meta)
            
            loss_d = criterion_disease(logits_d, labels_d)
            loss_c = criterion_concept(logits_c, labels_c)
            
            # SỰ TÁC ĐỘNG CỦA ALPHA NẰM Ở ĐÂY
            loss = loss_d + alpha * loss_c
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f}")
        
    # 3. Đánh giá trên Validation
    model.eval()
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            c_img = batch['clinic_img'].to(device)
            d_img = batch['derm_img'].to(device)
            meta = batch['metadata'].to(device)
            
            logits_d, logits_c = model(c_img, d_img, meta_features=meta)
            
            # Bệnh lý
            preds_d = torch.argmax(torch.softmax(logits_d, dim=1), dim=1).cpu().numpy()
            all_preds_d.extend(preds_d)
            all_labels_d.extend(batch['label_disease'].numpy())
            
            # Concept (Ngưỡng 0.5)
            preds_c = (torch.sigmoid(logits_c) > 0.5).int().cpu().numpy()
            all_preds_c.extend(preds_c)
            all_labels_c.extend(batch['concept_labels'].numpy())
            
    f1_disease = f1_score(all_labels_d, all_preds_d, average='macro', zero_division=0)
    f1_concept = f1_score(all_labels_c, all_preds_c, average='macro', zero_division=0)
    
    print(f">> KẾT QUẢ ALPHA = {alpha} | Disease F1: {f1_disease:.4f} | Concept F1: {f1_concept:.4f}")
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
    
    alphas = [0.25, 0.5, 1.0, 2.0]
    results = []
    
    print("="*60)
    print("BẮT ĐẦU ALPHA ABLATION STUDY CHO PROPOSED_HYBRID")
    print("="*60)
    
    for alpha in alphas:
        f1_d, f1_c = train_and_eval_alpha(alpha, train_loader, val_loader, device, dynamic_meta_dim)
        results.append({
            "Alpha": alpha,
            "Disease Macro-F1 (Val)": f"{f1_d:.4f}",
            "Concept Macro-F1 (Val)": f"{f1_c:.4f}"
        })
        
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BẢNG TỔNG HỢP ALPHA ABLATION")
    print("="*60)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    main()