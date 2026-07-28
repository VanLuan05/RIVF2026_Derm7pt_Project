import os
from random import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Nhập đầy đủ các module, bổ sung calculate_dataset_weights
from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms, calculate_dataset_weights
from src.models.models import MultimodalDermModel
from src.train import train_model

def main():
    # 1. Đường dẫn cấu hình (Tự động nhận diện Colab/Local)
    colab_drive_path = "/content/drive/MyDrive/RIVF2026_Dataset/data/"
    if os.path.exists(colab_drive_path):
        print("Phát hiện môi trường Google Colab...")
        base_dir = colab_drive_path
    else:
        print("Phát hiện môi trường máy tính cá nhân...")
        base_dir = "data/"
        
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    
    # Đường dẫn file Từ điển nhãn mới tạo
    LABEL_MAPPING_JSON = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    # 2. Khởi tạo Dataset (Chỉ load 1 lần để tiết kiệm RAM/Thời gian)
    print("Đang nạp dữ liệu (Bài toán 5 Lớp Derm7pt Chuẩn)...")
    train_dataset = MultimodalDermDataset(
        csv_file=TRAIN_CSV, img_dir=IMG_DIR, 
        label_mapping_path=LABEL_MAPPING_JSON, transform=train_transforms
    )
    val_dataset = MultimodalDermDataset(
        csv_file=VAL_CSV, img_dir=IMG_DIR, 
        label_mapping_path=LABEL_MAPPING_JSON, transform=test_transforms
    )
    
    workers = 2 if os.path.exists(colab_drive_path) else 0 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=workers)
    
    # 3. Tính toán tự động Class Weights và Pos Weights
    disease_weights, concept_pos_weights = calculate_dataset_weights(TRAIN_CSV, LABEL_MAPPING_JSON)
    num_disease_classes = len(train_dataset.disease_to_idx)
    print(f"Số lượng lớp bệnh (Disease Classes): {num_disease_classes}")
    
    # =====================================================================
    # VÒNG LẶP HUẤN LUYỆN KIỂM ĐỊNH TÍNH BỀN VỮNG (ROBUSTNESS)
    # =====================================================================
    random_seeds = [42, 100, 2026] # 3 hạt giống ngẫu nhiên để đánh giá
    
    for current_seed in random_seeds:
        print("\n" + "="*60)
        print(f"ĐANG KHỞI CHẠY THỰC NGHIỆM VỚI SEED: {current_seed}")
        print("="*60)
        
        # 4. Khóa hạt giống (Set Seed) để đảm bảo tính tái lập (Reproducibility)
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
            # Ép CuDNN tính toán một cách tất định (Deterministic)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False
            
        # 5. Khởi tạo LẠI mô hình (Để reset trọng số random ban đầu cho mỗi seed)
        model = MultimodalDermModel(
            num_classes=num_disease_classes, 
            num_concepts=7, 
            modality='dual', 
            bottleneck_type='multitask', # Dùng kiến trúc Master Model P2
            use_metadata=True
        )
        
        # Tạo tên file lưu trữ riêng biệt cho từng vòng lặp
        exp_name = f"best_model_P2_seed_{current_seed}"
        print(f"File trọng số sẽ được lưu với tên: {exp_name}.pth")
        
        # 6. Kích hoạt huấn luyện
        train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            disease_weights=disease_weights, 
            concept_pos_weights=concept_pos_weights, 
            num_epochs=20, 
            learning_rate=5e-5,
            experiment_name=exp_name  # <--- CHÚ Ý QUAN TRỌNG
        )
        
        print(f"ĐÃ HOÀN THÀNH HUẤN LUYỆN CHO SEED: {current_seed}")

if __name__ == "__main__":
    main()