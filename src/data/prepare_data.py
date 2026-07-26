import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def map_to_5_standard_classes(diagnosis):
    """Lỗi 2: Nhóm 20 bệnh thô về 5 nhóm chuẩn của Derm7pt"""
    diagnosis = str(diagnosis).lower().strip()
    
    if 'melanoma' in diagnosis:
        return 'Melanoma'
    elif 'nevus' in diagnosis or 'nevi' in diagnosis:
        return 'Nevus'
    elif 'basal cell carcinoma' in diagnosis:
        return 'Basal Cell Carcinoma'
    elif 'seborrheic keratosis' in diagnosis:
        return 'Seborrheic Keratosis'
    else:
        return 'Miscellaneous'

def encode_binary_abnormal_concept(concept_name, value):
    """Lỗi 3: Mã hóa Concept giữ lại thông tin lâm sàng (Abnormal = 1)"""
    val = str(value).lower().strip()
    if 'absent' in val or val == 'nan':
        return 0.0
        
    # Tùy chỉnh theo từng loại Concept để bắt đúng tính chất "bất thường"
    if concept_name in ['pigment_network']:
        return 1.0 if 'atypical' in val else 0.0
    elif concept_name in ['streaks']:
        return 1.0 if 'irregular' in val else 0.0
    elif concept_name in ['dots_and_globules']:
        return 1.0 if 'irregular' in val else 0.0
    # Các concept còn lại, cứ present là tính bất thường
    else:
        return 1.0

def main():
    print("Bắt đầu chuẩn hóa dữ liệu theo chuẩn Derm7pt...")
    # 1. Đọc file metadata gốc (bạn thay bằng đường dẫn file CSV gốc của bạn)
    raw_csv_path = "data/raw/meta/meta.csv" 
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(raw_csv_path)
    
    # [QUAN TRỌNG] Lọc bỏ các case bị lỗi file thật sự (Xóa cơ chế ảnh đen)
    # Giả sử bạn đã có hàm kiểm tra file tồn tại, hãy filter df ở đây.
    
    # 2. Xử lý Lỗi 2: Áp dụng 5 nhóm bệnh
    df['standard_diagnosis'] = df['diagnosis'].apply(map_to_5_standard_classes)
    
    # 3. Xử lý Lỗi 3: Áp dụng mã hóa Concept Abnormal
    concept_cols = [
        'pigment_network', 'streaks', 'pigmentation', 
        'regression_structures', 'dots_and_globules', 
        'blue_whitish_veil', 'vascular_structures'
    ]
    for col in concept_cols:
        df[f'{col}_encoded'] = df[col].apply(lambda x: encode_binary_abnormal_concept(col, x))

    # 4. Xử lý Lỗi 1: Khởi tạo Label Mapping chung cho toàn bộ dự án
    unique_diseases = sorted(df['standard_diagnosis'].unique())
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(disease_to_idx, f)
    print(f"Đã lưu từ điển nhãn: {disease_to_idx}")

    # 5. Xử lý Lỗi 7: Stratified Split (Chia dữ liệu bảo toàn tỷ lệ lớp)
    # Lấy nhãn để phân tầng
    labels = df['standard_diagnosis']
    
    # Chia Train (60%) và Rest (40%)
    train_df, rest_df = train_test_split(df, test_size=0.4, stratify=labels, random_state=42)
    
    # Chia Rest thành Validation (15%), Calibration (10%), Test (15%)
    rest_labels = rest_df['standard_diagnosis']
    val_df, temp_df = train_test_split(rest_df, test_size=0.625, stratify=rest_labels, random_state=42) # 0.625 * 40 = 25
    
    temp_labels = temp_df['standard_diagnosis']
    calib_df, test_df = train_test_split(temp_df, test_size=0.6, stratify=temp_labels, random_state=42) # 0.6 * 25 = 15
    
    # 6. Lưu file
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    calib_df.to_csv(os.path.join(output_dir, "calib_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"Hoàn tất! Số lượng mẫu: Train={len(train_df)}, Val={len(val_df)}, Calib={len(calib_df)}, Test={len(test_df)}")
    
if __name__ == "__main__":
    main()