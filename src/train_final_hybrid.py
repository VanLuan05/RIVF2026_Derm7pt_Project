import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import joblib
import random
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms
from src.models.models import MultimodalDermModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_final_hybrid(seed, alpha, train_loader, val_loader, device, dynamic_meta_dim, output_dir):
    set_seed(seed)
    print(f"\n{'='*50}")
    print(f"BẮT ĐẦU HUẤN LUYỆN PROPOSED_HYBRID | SEED: {seed} | ALPHA: {alpha}")
    print(f"{'='*50}")
    
    model = MultimodalDermModel(
        num_classes=5, num_concepts=7, 
        modality='dual', bottleneck_type='hybrid', 
        use_metadata=True, meta_input_dim=dynamic_meta_dim
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) # Có weight_decay chống overfitting
    criterion_disease = nn.CrossEntropyLoss()
    criterion_concept = nn.BCEWithLogitsLoss()
    
    epochs = 20
    best_val_f1 = -1.0
    model_save_path = os.path.join(output_dir, f"Proposed_Hybrid_seed_{seed}.pth")
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # 1. Huấn luyện (Train)
        model.train()
        train_loss = 0.0
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
            train_loss += loss.item()
            
        # 2. Đánh giá (Validation) để chọn Model tốt nhất
        model.eval()
        all_preds_d, all_labels_d = [], []
        with torch.no_grad():
            for batch in val_loader:
                c_img, d_img = batch['clinic_img'].to(device), batch['derm_img'].to(device)
                meta = batch['metadata'].to(device)
                logits_d, _ = model(c_img, d_img, meta_features=meta)
                
                preds_d = torch.argmax(torch.softmax(logits_d, dim=1), dim=1).cpu().numpy()
                all_preds_d.extend(preds_d)
                all_labels_d.extend(batch['label_disease'].numpy())
                
        val_f1 = f1_score(all_labels_d, all_preds_d, average='macro', zero_division=0)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Disease F1: {val_f1:.4f}")
        
        # 3. Lưu trọng số nếu Validation F1 tăng (Model Selection)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  [+] Đã lưu checkpoint mới tốt nhất: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        # 4. Early Stopping
        if epochs_no_improve >= patience:
            print(f"  [-] Dừng sớm tại Epoch {epoch+1} do Validation F1 không tăng trong {patience} epochs liên tiếp.")
            break
            
    print(f"KẾT THÚC SEED {seed} | BEST VAL F1: {best_val_f1:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    IMG_DIR = "/content/local_images/" if os.path.exists("/content/local_images/") else os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    
    try:
        encoder = joblib.load(os.path.join(OUTPUT_DIR, "meta_encoder.joblib"))
        dynamic_meta_dim = len(encoder.get_feature_names_out())
    except:
        dynamic_meta_dim = 14
        
    train_dataset = MultimodalDermDataset(TRAIN_CSV, IMG_DIR, LABEL_MAPPING, transform=train_transforms)
    val_dataset = MultimodalDermDataset(VAL_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    
    workers = 2 if "content" in base_dir else 0
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    # CHỐT SỔ SIÊU THAM SỐ
    final_alpha = 3.0
    seeds = [42, 100, 2026]
    
    for seed in seeds:
        train_final_hybrid(seed, final_alpha, train_loader, val_loader, device, dynamic_meta_dim, OUTPUT_DIR)
        
    print("\n" + "*"*60)
    print("ĐÃ HOÀN TẤT HUẤN LUYỆN PROPOSED_HYBRID.")
    print("Mô hình đã được lưu đè vào thư mục outputs/. Hãy chạy file run_evaluation.py để xem kết quả trên tập Test!")
    print("*"*60)

if __name__ == "__main__":
    main()