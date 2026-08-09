import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import torch
import numpy as np
import random
import joblib
from torch.utils.data import DataLoader
from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms, calculate_dataset_weights
from src.models.models import MultimodalDermModel
from src.train import train_model

def run_experiment(exp_name, modality, bottleneck_type, use_metadata, meta_input_dim, seed):
    print("\n" + "="*60)
    print(f"ĐANG CHẠY THỰC NGHIỆM: {exp_name} (Seed: {seed})")
    print(f"Cấu hình: Modality={modality}, Bottleneck={bottleneck_type}, Metadata={use_metadata}")
    print("="*60)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    # Bổ sung truyền biến meta_input_dim vào mô hình
    model = MultimodalDermModel(
        num_classes=5,       
        num_concepts=7,      
        modality=modality, 
        bottleneck_type=bottleneck_type, 
        use_metadata=use_metadata,
        meta_input_dim=meta_input_dim # <--- Dynamic Metadata Dimension
    )
    
    save_name = f"{exp_name}_seed_{seed}"
    print(f"Sẽ lưu trọng số tại: outputs/{save_name}.pth")
    
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
    base_dir = "/content/drive/MyDrive/RIVF2026_Dataset/data/" if os.path.exists("/content/drive/MyDrive/RIVF2026_Dataset/data/") else "data/"
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = "/content/local_images/" if os.path.exists("/content/local_images/") else os.path.join(base_dir, "raw/images/")
    
    train_dataset = MultimodalDermDataset(TRAIN_CSV, IMG_DIR, LABEL_MAPPING, transform=train_transforms)
    val_dataset = MultimodalDermDataset(VAL_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    
    workers = 2 if "content" in base_dir else 0 
    global train_loader, val_loader, disease_weights, concept_pos_weights
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    disease_weights, concept_pos_weights = calculate_dataset_weights(TRAIN_CSV, LABEL_MAPPING)

    # Đọc số chiều Metadata tự động từ khuôn
    try:
        encoder = joblib.load("outputs/meta_encoder.joblib")
        dynamic_meta_dim = len(encoder.get_feature_names_out())
        print(f"[*] Số chiều Metadata tự động nhận diện: {dynamic_meta_dim}")
    except Exception as e:
        print(f"[!] Không tìm thấy meta_encoder.joblib, dùng mặc định 14 chiều. Lỗi: {e}")
        dynamic_meta_dim = 14

    # KỊCH BẢN CHẠY TỰ ĐỘNG CHUẨN MỚI
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

    for exp in experiments:
        for s in seeds:
            # Bổ sung dynamic_meta_dim vào vòng lặp gọi hàm
            run_experiment(exp["name"], exp["modality"], exp["bottleneck"], exp["meta"], dynamic_meta_dim, s)
            
    print("TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN ABLATION ĐÃ HOÀN TẤT!")