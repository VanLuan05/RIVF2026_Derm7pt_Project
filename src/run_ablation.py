import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms, calculate_dataset_weights
from src.models.models import MultimodalDermModel
from src.train import train_model

def run_experiment(exp_name, modality, bottleneck_type, use_metadata, seed):
    print("\n" + "="*60)
    print(f"ĐANG CHẠY THỰC NGHIỆM: {exp_name} (Seed: {seed})")
    print(f"Cấu hình: Modality={modality}, Bottleneck={bottleneck_type}, Metadata={use_metadata}")
    print("="*60)
    
    # Khóa Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    # Khởi tạo mô hình
    model = MultimodalDermModel(
        num_classes=5,       # 5 lớp bệnh chuẩn
        num_concepts=7,      # 7 khái niệm lâm sàng
        modality=modality, 
        bottleneck_type=bottleneck_type, 
        use_metadata=use_metadata
    )
    
    # Checkpoint name
    save_name = f"{exp_name}_seed_{seed}"
    print(f"Sẽ lưu trọng số tại: outputs/{save_name}.pth")
    
    # Train
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        disease_weights=disease_weights, 
        concept_pos_weights=concept_pos_weights, 
        num_epochs=20, 
        learning_rate=5e-5,
        experiment_name=save_name
    )
    print(f"Hoàn thành: {exp_name}\n")

if __name__ == "__main__":
    # 1. Đường dẫn cấu hình
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    # 2. Chuẩn bị DataLoader (Chỉ load 1 lần)
    train_dataset = MultimodalDermDataset(TRAIN_CSV, IMG_DIR, LABEL_MAPPING, transform=train_transforms)
    val_dataset = MultimodalDermDataset(VAL_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    
    workers = 2 if "content" in base_dir else 0 
    global train_loader, val_loader, disease_weights, concept_pos_weights
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    disease_weights, concept_pos_weights = calculate_dataset_weights(TRAIN_CSV, LABEL_MAPPING)

    # 3. KỊCH BẢN CHẠY TỰ ĐỘNG B1-B5 VÀ P2 (Cấu hình chuẩn Ablation Study)
    experiments = [
        {"name": "B1_Clinical_Only", "modality": "clinic_only", "bottleneck": "none", "meta": False},
        {"name": "B2_Derm_Only",     "modality": "derm_only",   "bottleneck": "none", "meta": False},
        {"name": "B3_Meta_Only",     "modality": "meta_only",   "bottleneck": "none", "meta": True},
        {"name": "B4_Dual_NoMeta",   "modality": "dual",        "bottleneck": "none", "meta": False},
        {"name": "B5_Dual_PureCBM",  "modality": "dual",        "bottleneck": "pure", "meta": True},
        {"name": "Master_P2",        "modality": "dual",        "bottleneck": "multitask", "meta": True}
    ]

    seeds = [42, 100, 2026] # Theo chuẩn P1 của thầy

    # Chạy vòng lặp tự động (Chạy hàng loạt)
    for exp in experiments:
        for s in seeds:
            run_experiment(exp["name"], exp["modality"], exp["bottleneck"], exp["meta"], s)
            
    print("TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ABLATION ĐÃ HOÀN TẤT!")