import torch
import os # Thêm thư viện os để kiểm tra đường dẫn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms
from src.models.models import MultimodalDermModel
from src.train import train_model

def main():
    # 1. Tự động phát hiện môi trường để chọn đúng đường dẫn
    colab_drive_path = "/content/drive/MyDrive/RIVF2026_Dataset/data/"
    
    if os.path.exists(colab_drive_path):
        print("Phát hiện môi trường Google Colab, đang lấy dữ liệu từ Google Drive...")
        base_dir = colab_drive_path
    else:
        print("Phát hiện môi trường máy tính cá nhân, đang lấy dữ liệu cục bộ...")
        base_dir = "data/"
        
    TRAIN_CSV = os.path.join(base_dir, "processed/train_split.csv")
    VAL_CSV = os.path.join(base_dir, "processed/val_split.csv")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    
    # 2. Khởi tạo Dataset
    print("Đang nạp dữ liệu...")
    # ... (Giữ nguyên toàn bộ phần code phía dưới của bạn) ...
    train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=train_transforms)
    val_dataset = MultimodalDermDataset(csv_file=VAL_CSV, img_dir=IMG_DIR, transform=test_transforms)
    
    num_disease_classes = len(train_dataset.disease_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # Đổi num_workers=2 cho ổn định trên Colab
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Khởi tạo mô hình dự đoán {num_disease_classes} bệnh và 7 khái niệm...")
    model = MultimodalDermModel(num_classes=num_disease_classes, num_concepts=7, use_metadata=False) 
    
    print("Bắt đầu vòng lặp huấn luyện...")
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4)

if __name__ == "__main__":
    main()