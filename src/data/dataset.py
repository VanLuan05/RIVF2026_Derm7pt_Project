import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# 1. Định nghĩa các phép biến đổi ảnh (Transforms)
# Dùng kích thước 224x224 chuẩn của ResNet/EfficientNet
# Chuẩn hóa (Normalize) theo ImageNet để Transfer Learning hiệu quả

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Lật ngang ngẫu nhiên
    transforms.RandomRotation(15),     # Xoay nhẹ để tạo tính đa dạng
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test/Val/Cal chỉ được Resize và Normalize, KHÔNG lật hay xoay
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultimodalDermDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file: Đường dẫn tới file train_split.csv, val_split.csv...
        img_dir: Đường dẫn tới thư mục data/raw/images/
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Tạo một từ điển map tên bệnh (chuỗi) sang số nguyên (int)
        # (Ở dự án thực tế, bạn sẽ dùng LabelEncoder của sklearn để map tự động)
        unique_diseases = self.data_frame['diagnosis'].unique()
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        clinic_path = os.path.join(self.img_dir, str(row['clinic']))
        derm_path = os.path.join(self.img_dir, str(row['derm']))

        clinic_img = Image.open(clinic_path).convert('RGB')
        derm_img = Image.open(derm_path).convert('RGB')

        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        meta_features = {
            'sex': str(row['sex']),
            'location': str(row['location']),
            'elevation': str(row['elevation'])
        }

        target_disease = self.disease_to_idx[row['diagnosis']]
        
        # --- BẮT ĐẦU PHẦN CẬP NHẬT: SỐ HÓA 7 KHÁI NIỆM ---
        # 1. Định nghĩa 7 cột khái niệm (dựa theo tiêu đề cột trong meta.csv)
        concept_cols = [
            'pigment_network', 'streaks', 'pigmentation', 
            'regression_structures', 'dots_and_globules', 
            'blue_whitish_veil', 'vascular_structures'
        ]
        
        # 2. Xử lý chuyển đổi chữ thành số (0 hoặc 1)
        concept_values = []
        for col in concept_cols:
            val = str(row[col]).lower().strip()
            # Nếu chứa từ 'absent' hoặc NaN -> 0, ngược lại (có triệu chứng) -> 1
            if 'absent' in val or val == 'nan':
                concept_values.append(0.0)
            else:
                concept_values.append(1.0)
                
        # 3. Chuyển list Python thành Tensor của PyTorch
        concept_tensor = torch.tensor(concept_values, dtype=torch.float)
        # --- KẾT THÚC PHẦN CẬP NHẬT ---

        sample = {
            'clinic_img': clinic_img,
            'derm_img': derm_img,
            'metadata': meta_features,
            'label_disease': torch.tensor(target_disease, dtype=torch.long),
            'concept_labels': concept_tensor # Đã đổi tên và dạng dữ liệu
        }

        return sample

if __name__ == "__main__":
    # Đường dẫn (hãy đảm bảo thư mục đang đứng là thư mục gốc của project)
    TRAIN_CSV = "data/processed/train_split.csv"
    IMG_DIR = "data/raw/images/"
    
    # Khởi tạo Dataset
    train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=train_transforms)
    
    # Khởi tạo DataLoader (Nạp theo cụm 16 hoặc 32 ca bệnh một lúc)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Lấy thử 1 batch ra kiểm tra
    for batch in train_loader:
        print("Đã load thành công 1 Batch!")
        print(f"- Kích thước tensor Ảnh Clinic: {batch['clinic_img'].shape}") 
        print(f"- Kích thước tensor Ảnh Derm:   {batch['derm_img'].shape}")
        print(f"- Nhãn bệnh (Diagnosis IDs):    {batch['label_disease']}")
        print(f"- Metadata (Giới tính):         {batch['metadata']['sex']}")
        break # Chỉ lấy 1 batch để test rồi dừng