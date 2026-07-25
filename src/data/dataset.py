import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ==========================================
# 1. Định nghĩa các phép biến đổi ảnh (Transforms)
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
    """
    Hàm dò tìm đường dẫn thực tế trên Linux bỏ qua sự khác biệt hoa/thường.
    Ví dụ: 'FCl/Fcl068.jpg' sẽ tự khớp với thư mục 'FCL/fcl068.jpg' thực tế.
    """
    # Chuẩn hóa dấu gạch chéo
    parts = str(relative_path).replace('\\', '/').split('/')
    current_path = base_dir
    
    for part in parts:
        # Nếu đường dẫn hiện tại không tồn tại thì dừng
        if not os.path.exists(current_path):
            return None
            
        # Lấy danh sách tất cả file/thư mục thực tế đang có
        actual_items = os.listdir(current_path)
        
        # Tạo từ điển ánh xạ: tên chữ thường -> tên thật
        lower_map = {item.lower(): item for item in actual_items}
        
        part_lower = part.lower()
        # Nếu tìm thấy thư mục/file khớp ở dạng chữ thường, nối tên thật vào đường dẫn
        if part_lower in lower_map:
            current_path = os.path.join(current_path, lower_map[part_lower])
        else:
            return None # File thực sự bị mất khỏi Drive
            
    return current_path

# ==========================================
# 3. Trái Tim Của Việc Nạp Dữ Liệu (Dataset Class)
# ==========================================
class MultimodalDermDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Tạo từ điển map tên bệnh (chuỗi) sang số nguyên (int)
        unique_diseases = self.data_frame['diagnosis'].unique()
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # SỬ DỤNG HÀM DÒ TÌM THÔNG MINH ĐỂ LẤY ĐƯỜNG DẪN THẬT
        clinic_path = get_real_path(self.img_dir, row['clinic'])
        derm_path = get_real_path(self.img_dir, row['derm'])

        # BỌC ÁO GIÁP CHO ẢNH CLINIC
        try:
            if clinic_path is None:
                raise FileNotFoundError
            clinic_img = Image.open(clinic_path).convert('RGB')
        except FileNotFoundError:
            print(f"\n[Cảnh báo] Mất file thật sự: {row['clinic']}")
            # Tạo ảnh đen rỗng để mô hình không bị ngắt quãng
            clinic_img = Image.new('RGB', (224, 224), (0, 0, 0))

        # BỌC ÁO GIÁP CHO ẢNH DERMOSCOPY
        try:
            if derm_path is None:
                raise FileNotFoundError
            derm_img = Image.open(derm_path).convert('RGB')
        except FileNotFoundError:
            print(f"\n[Cảnh báo] Mất file thật sự: {row['derm']}")
            derm_img = Image.new('RGB', (224, 224), (0, 0, 0))

        # Áp dụng biến đổi (Transforms)
        if self.transform:
            clinic_img = self.transform(clinic_img)
            derm_img = self.transform(derm_img)

        # Trích xuất Metadata cơ bản
        meta_features = {
            'sex': str(row['sex']),
            'location': str(row['location']),
            'elevation': str(row['elevation'])
        }

        # Lấy nhãn bệnh (Dạng số nguyên)
        target_disease = self.disease_to_idx[row['diagnosis']]
        
        # SỐ HÓA 7 KHÁI NIỆM (CONCEPTS BOTTLE-NECK)
        concept_cols = [
            'pigment_network', 'streaks', 'pigmentation', 
            'regression_structures', 'dots_and_globules', 
            'blue_whitish_veil', 'vascular_structures'
        ]
        
        concept_values = []
        for col in concept_cols:
            val = str(row[col]).lower().strip()
            # Quy ước: Nếu absent (không có) hoặc rỗng thì bằng 0, ngược lại bằng 1
            if 'absent' in val or val == 'nan':
                concept_values.append(0.0)
            else:
                concept_values.append(1.0)
                
        # Chuyển thành Tensor cho hàm loss
        concept_tensor = torch.tensor(concept_values, dtype=torch.float)

        # Trả về một từ điển hoàn chỉnh chứa mọi luồng dữ liệu của 1 ca bệnh
        sample = {
            'clinic_img': clinic_img,
            'derm_img': derm_img,
            'metadata': meta_features,
            'label_disease': torch.tensor(target_disease, dtype=torch.long),
            'concept_labels': concept_tensor
        }

        return sample

# ==========================================
# 4. Kiểm Thử Nhanh (Sanity Check)
# ==========================================
if __name__ == "__main__":
    TRAIN_CSV = "data/processed/train_split.csv"
    IMG_DIR = "data/raw/images/"
    
    # Kiểm tra xem đường dẫn cục bộ có tồn tại không để test
    if os.path.exists(TRAIN_CSV):
        train_dataset = MultimodalDermDataset(csv_file=TRAIN_CSV, img_dir=IMG_DIR, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        
        for batch in train_loader:
            print("Đã load thành công 1 Batch với cấu hình Dataset hoàn chỉnh!")
            print(f"- Kích thước tensor Ảnh Clinic: {batch['clinic_img'].shape}") 
            print(f"- Kích thước tensor Ảnh Derm:   {batch['derm_img'].shape}")
            print(f"- Nhãn bệnh (Diagnosis IDs):    {batch['label_disease']}")
            print(f"- Nhãn khái niệm (Concepts):    {batch['concept_labels'].shape}")
            break
    else:
        print("Không tìm thấy dữ liệu để test cục bộ. Vui lòng chạy trên Google Colab qua file main.py!")