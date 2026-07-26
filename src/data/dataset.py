import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. Các phép biến đổi ảnh (Transforms)
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. Hàm Dò Tìm Đường Dẫn Thông Minh (Chống lỗi Hoa/Thường)
# ==========================================
def get_real_path(base_dir, relative_path):
    parts = str(relative_path).replace('\\', '/').split('/')
    current_path = base_dir
    
    for part in parts:
        if not os.path.exists(current_path):
            return None
        actual_items = os.listdir(current_path)
        lower_map = {item.lower(): item for item in actual_items}
        
        part_lower = part.lower()
        if part_lower in lower_map:
            current_path = os.path.join(current_path, lower_map[part_lower])
        else:
            return None 
            
    return current_path

# ==========================================
# 3. Trái Tim Của Việc Nạp Dữ Liệu (Dataset Class)
# ==========================================
class MultimodalDermDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_mapping_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # ĐỌC MAPPING TỪ FILE JSON (Khắc phục Lỗi 1)
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file Mapping tại: {label_mapping_path}. Hãy chạy prepare_data.py trước!")
            
        with open(label_mapping_path, 'r') as f:
            self.disease_to_idx = json.load(f)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # SỬ DỤNG HÀM DÒ TÌM THÔNG MINH
        clinic_path = get_real_path(self.img_dir, row['clinic'])
        derm_path = get_real_path(self.img_dir, row['derm'])

        # GỠ BỎ HOÀN TOÀN ÁO GIÁP ẢNH ĐEN. Nếu thiếu file, văng lỗi ngay để dọn dẹp Data.
        if clinic_path is None or not os.path.exists(clinic_path):
            raise FileNotFoundError(f"Mất file Clinic thật sự tại dòng {idx}: {row['clinic']}")
        if derm_path is None or not os.path.exists(derm_path):
            raise FileNotFoundError(f"Mất file Derm thật sự tại dòng {idx}: {row['derm']}")
            
        clinic_img = Image.open(clinic_path).convert('RGB')
        derm_img = Image.open(derm_path).convert('RGB')

        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        # LẤY NHÃN BỆNH CHUẨN (Đã chuyển về 5 Lớp ở prepare_data.py)
        target_disease = self.disease_to_idx[row['standard_diagnosis']]
        
        # ĐỌC 7 KHÁI NIỆM ĐÃ ĐƯỢC MÃ HÓA (Abnormal = 1, Normal = 0) (Khắc phục Lỗi 3)
        concept_cols = [
            'pigment_network_encoded', 'streaks_encoded', 'pigmentation_encoded', 
            'regression_structures_encoded', 'dots_and_globules_encoded', 
            'blue_whitish_veil_encoded', 'vascular_structures_encoded'
        ]
        
        # Ép kiểu an toàn sang float
        concept_values = [float(row[col]) for col in concept_cols]
        concept_tensor = torch.tensor(concept_values, dtype=torch.float)

        meta_features = {
            'sex': str(row['sex']),
            'location': str(row['location']),
            'elevation': str(row['elevation'])
        }

        sample = {
            'clinic_img': clinic_img,
            'derm_img': derm_img,
            'metadata': meta_features,
            'label_disease': torch.tensor(target_disease, dtype=torch.long),
            'concept_labels': concept_tensor
        }

        return sample

# ==========================================
# 4. Tiện Ích: Tính Toán Trọng Số Dựa Trên Phân Phố Thực Tế (Khắc phục Lỗi 8)
# ==========================================
def calculate_dataset_weights(csv_file, label_mapping_path):
    """
    Hàm này chỉ nên chạy trên tập TRAIN để tính toán ra:
    1. disease_weights: Trọng số phạt cho 5 lớp bệnh (CrossEntropy)
    2. concept_pos_weights: Trọng số phạt cho 7 khái niệm (BCEWithLogitsLoss)
    Dựa trên đúng số lượng mẫu đếm được, tuyệt đối không đặt tay.
    """
    df = pd.read_csv(csv_file)
    with open(label_mapping_path, 'r') as f:
        disease_to_idx = json.load(f)
        
    print("\nĐANG TÍNH TOÁN TRỌNG SỐ TỰ ĐỘNG THEO DỮ LIỆU THỰC...")
    
    # --- 1. TÍNH TRỌNG SỐ BỆNH ---
    # Đếm số lượng từng bệnh theo index
    disease_counts = df['standard_diagnosis'].map(disease_to_idx).value_counts().sort_index().values
    total_samples = len(df)
    num_classes = len(disease_counts)
    
    # Công thức chuẩn: class_weight = total_samples / (num_classes * count)
    disease_weights = total_samples / (num_classes * disease_counts)
    print(f"Trọng số Bệnh (Disease Weights): {np.round(disease_weights, 4)}")
    
    # --- 2. TÍNH TRỌNG SỐ KHÁI NIỆM (POS WEIGHTS) ---
    concept_cols = [
        'pigment_network_encoded', 'streaks_encoded', 'pigmentation_encoded', 
        'regression_structures_encoded', 'dots_and_globules_encoded', 
        'blue_whitish_veil_encoded', 'vascular_structures_encoded'
    ]
    
    concept_pos_weights = []
    for col in concept_cols:
        positive_count = df[col].sum()
        negative_count = total_samples - positive_count
        
        # Nếu mẫu quá hiếm (positive_count = 0) để tránh lỗi chia cho 0
        if positive_count == 0:
             pos_weight = 1.0
        else:
             # Công thức chuẩn: pos_weight = negative_count / positive_count
             pos_weight = negative_count / positive_count
             
        concept_pos_weights.append(pos_weight)
        
    print(f"Trọng số Khái niệm (Concept Pos Weights): {np.round(concept_pos_weights, 4)}")
    
    return torch.tensor(disease_weights, dtype=torch.float), torch.tensor(concept_pos_weights, dtype=torch.float)