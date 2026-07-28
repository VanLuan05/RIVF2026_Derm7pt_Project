import os
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
    
    # 2. Khởi tạo Dataset (Cần truyền label_mapping_path vào)
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
    
    # 3. Tính toán tự động Class Weights và Pos Weights từ phân phối gốc (Lỗi 8)
    disease_weights, concept_pos_weights = calculate_dataset_weights(TRAIN_CSV, LABEL_MAPPING_JSON)
    
    num_disease_classes = len(train_dataset.disease_to_idx)
    print(f"Số lượng lớp bệnh (Disease Classes): {num_disease_classes}")
    
   # 4. Khởi tạo mô hình
    print("\n" + "="*50)
    print("ĐANG KHỞI TẠO MÔ HÌNH BASELINE")
    print("="*50)
    
    model = MultimodalDermModel(
        num_classes=num_disease_classes, 
        num_concepts=7, 
        modality='dual',  # 'dual' cho cả clinic + derm, 'meta_only' chỉ metadata, 'derm_only' chỉ derm
        bottleneck_type='hybrid',  
        use_metadata=True
    )
    
    # 5. Kích hoạt huấn luyện
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        disease_weights=disease_weights, 
        concept_pos_weights=concept_pos_weights, #train.py sẽ tự bỏ qua biến này
        num_epochs=20, 
        learning_rate=5e-5
    )

if __name__ == "__main__":
    main()