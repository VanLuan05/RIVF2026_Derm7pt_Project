import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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
# 2. Hàm Dò Tìm Đường Dẫn Thông Minh
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
# 3. (Dataset Class)
# ==========================================
class MultimodalDermDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_mapping_path, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"Không tìm thấy file Mapping tại: {label_mapping_path}")
            
        with open(label_mapping_path, 'r') as f:
            self.disease_to_idx = json.load(f)

        #  GIẢI QUYẾT TỪ DỮ LIỆU: Lấy danh sách các giá trị Metadata duy nhất
        self.sex_cats = self.data_frame['sex'].astype(str).unique().tolist()
        self.loc_cats = self.data_frame['location'].astype(str).unique().tolist()
        self.elev_cats = self.data_frame['elevation'].astype(str).unique().tolist()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        clinic_path = get_real_path(self.img_dir, row['clinic'])
        derm_path = get_real_path(self.img_dir, row['derm'])

        if clinic_path is None or not os.path.exists(clinic_path):
            raise FileNotFoundError(f"Mất file Clinic thật sự tại dòng {idx}")
        if derm_path is None or not os.path.exists(derm_path):
            raise FileNotFoundError(f"Mất file Derm thật sự tại dòng {idx}")
            
        clinic_img = Image.open(clinic_path).convert('RGB')
        derm_img = Image.open(derm_path).convert('RGB')

        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        target_disease = self.disease_to_idx[row['standard_diagnosis']]
        
        concept_cols = [
            'pigment_network_encoded', 'streaks_encoded', 'pigmentation_encoded', 
            'regression_structures_encoded', 'dots_and_globules_encoded', 
            'blue_whitish_veil_encoded', 'vascular_structures_encoded'
        ]
        concept_values = [float(row[col]) for col in concept_cols]
        concept_tensor = torch.tensor(concept_values, dtype=torch.float)

        # =========================================================
        # MÃ HÓA METADATA (ONE-HOT ENCODING)
        # =========================================================
        sex_vec = [1.0 if str(row['sex']) == c else 0.0 for c in self.sex_cats]
        loc_vec = [1.0 if str(row['location']) == c else 0.0 for c in self.loc_cats]
        elev_vec = [1.0 if str(row['elevation']) == c else 0.0 for c in self.elev_cats]
        
        # Nối tất cả thành 1 list
        meta_features_list = sex_vec + loc_vec + elev_vec
        
        # Đệm thêm số 0 cho đủ kích thước 32 chiều (khớp với model)
        if len(meta_features_list) < 32:
            meta_features_list += [0.0] * (32 - len(meta_features_list))
        else:
            meta_features_list = meta_features_list[:32]
            
        meta_tensor = torch.tensor(meta_features_list, dtype=torch.float)

        sample = {
            'clinic_img': clinic_img,
            'derm_img': derm_img,
            'metadata': meta_tensor, # <--- Trả về Tensor 32 chiều, không trả String nữa
            'label_disease': torch.tensor(target_disease, dtype=torch.long),
            'concept_labels': concept_tensor
        }

        return sample

# (Giữ nguyên hàm calculate_dataset_weights ở dưới cùng)
def calculate_dataset_weights(csv_file, label_mapping_path):
    df = pd.read_csv(csv_file)
    with open(label_mapping_path, 'r') as f:
        disease_to_idx = json.load(f)
        
    print("\nĐANG TÍNH TOÁN TRỌNG SỐ TỰ ĐỘNG THEO DỮ LIỆU THỰC...")
    
    disease_counts = df['standard_diagnosis'].map(disease_to_idx).value_counts().sort_index().values
    total_samples = len(df)
    num_classes = len(disease_counts)
    
    disease_weights = total_samples / (num_classes * disease_counts)
    print(f"Trọng số Bệnh (Disease Weights): {np.round(disease_weights, 4)}")
    
    concept_cols = [
        'pigment_network_encoded', 'streaks_encoded', 'pigmentation_encoded', 
        'regression_structures_encoded', 'dots_and_globules_encoded', 
        'blue_whitish_veil_encoded', 'vascular_structures_encoded'
    ]
    
    concept_pos_weights = []
    for col in concept_cols:
        positive_count = df[col].sum()
        negative_count = total_samples - positive_count
        
        if positive_count == 0:
             pos_weight = 1.0
        else:
             pos_weight = negative_count / positive_count
             
        concept_pos_weights.append(pos_weight)
        
    print(f"Trọng số Khái niệm (Concept Pos Weights): {np.round(concept_pos_weights, 4)}")
    return torch.tensor(disease_weights, dtype=torch.float), torch.tensor(concept_pos_weights, dtype=torch.float)