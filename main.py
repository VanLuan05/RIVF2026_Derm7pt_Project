import torch
from torch.utils.data import DataLoader
from torchvision import transforms

## main.py: Tập hợp các thành phần để chạy huấn luyện mô hình
from src.data.dataset import MultimodalDermDataset, train_transforms, test_transforms
from src.models.models import MultimodalDermModel
from src.train import train_model

def main():
    # 1. Đường dẫn file
    TRAIN_CSV = "data/processed/train_split.csv"
    VAL_CSV = "data/processed/val_split.csv"
    IMG_DIR = "data/raw/images/"
    
    # 2. Khởi tạo Dataset
    print("Đang nạp dữ liệu...")
    train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=train_transforms)
    val_dataset = MultimodalDermDataset(csv_file=VAL_CSV, img_dir=IMG_DIR, transform=test_transforms)
    
    # Đồng bộ từ điển bệnh (Đảm bảo mô hình biết có bao nhiêu lớp bệnh)
    num_disease_classes = len(train_dataset.disease_to_idx)
    
    # 3. Khởi tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 4. Khởi tạo Mô hình AI
    print(f"Khởi tạo mô hình dự đoán {num_disease_classes} bệnh và 7 khái niệm...")
    # Tạm tắt nhánh metadata (use_metadata=False) để test luồng ảnh trước cho nhẹ
    model = MultimodalDermModel(num_classes=num_disease_classes, num_concepts=7, use_metadata=False) 
    
    # 5. Kích hoạt huấn luyện
    print("Bắt đầu vòng lặp huấn luyện...")
    train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4)

if __name__ == "__main__":
    main()