import pandas as pd
import json
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

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
    """SỬA LỖI P0: Mã hóa Concept giữ lại thông tin lâm sàng chuẩn xác"""
    val = str(value).lower().strip()
    if 'absent' in val or val == 'nan' or val == 'none':
        return 0.0
        
    # Tùy chỉnh theo từng loại Concept để bắt đúng tính chất "bất thường"
    if concept_name in ['pigment_network']:
        return 1.0 if 'atypical' in val else 0.0
    elif concept_name in ['streaks', 'pigmentation', 'dots_and_globules', 'vascular_structures']:
        return 1.0 if 'irregular' in val else 0.0
    elif concept_name in ['blue_whitish_veil', 'regression_structures']:
        return 1.0 if 'present' in val else 0.0
    else:
        return 0.0

def main():
    print("Bắt đầu chuẩn hóa dữ liệu theo chuẩn Derm7pt...")
    
    # =================================================================
    # CẬP NHẬT ĐƯỜNG DẪN: Tự động nhận diện Google Drive trên Colab
    # =================================================================
    colab_drive_path = "/content/drive/MyDrive/RIVF2026_Dataset/data/"
    if os.path.exists(colab_drive_path):
        print("Phát hiện môi trường Google Colab, đang trỏ tới Google Drive...")
        base_dir = colab_drive_path
    else:
        print("Phát hiện môi trường máy tính cá nhân...")
        base_dir = "data/"
        
    raw_csv_path = os.path.join(base_dir, "raw/meta/meta.csv")
    output_dir = os.path.join(base_dir, "processed/")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/", exist_ok=True) # Thư mục outputs nằm ở project root
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {raw_csv_path}. Hãy kiểm tra lại Google Drive!")
        
    df = pd.read_csv(raw_csv_path)
    
    # 1. Xử lý Lỗi 2: Áp dụng 5 nhóm bệnh
    df['standard_diagnosis'] = df['diagnosis'].apply(map_to_5_standard_classes)
    
    # 2. Xử lý Lỗi P0 (Mục 2): Áp dụng mã hóa Concept Abnormal chuẩn y khoa
    concept_cols = [
        'pigment_network', 'streaks', 'pigmentation', 
        'regression_structures', 'dots_and_globules', 
        'blue_whitish_veil', 'vascular_structures'
    ]
    for col in concept_cols:
        df[f'{col}_encoded'] = df[col].apply(lambda x: encode_binary_abnormal_concept(col, x))

    # 3. Khởi tạo Label Mapping chung cho toàn bộ dự án
    unique_diseases = sorted(df['standard_diagnosis'].unique())
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(disease_to_idx, f)
    print(f"Đã lưu từ điển nhãn: {disease_to_idx}")

    # 4. SỬA LỖI P0 (Mục 3): Patient-level Split (Chống rò rỉ dữ liệu)
    df['case_num'] = df['case_num'].fillna('unknown_case')
    
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    rest_idx, test_idx = next(gss_test.split(df, groups=df['case_num']))
    rest_df = df.iloc[rest_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    gss_calib = GroupShuffleSplit(n_splits=1, test_size=0.1176, random_state=42)
    train_val_idx, calib_idx = next(gss_calib.split(rest_df, groups=rest_df['case_num']))
    train_val_df = rest_df.iloc[train_val_idx].copy()
    calib_df = rest_df.iloc[calib_idx].copy()
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['case_num']))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # 5. SỬA LỖI P0 (Mục 1): Metadata Encoder dùng chung
    print("Đang huấn luyện (Fit) Metadata Encoder duy nhất trên tập TRAIN...")
    meta_cols = ['sex', 'location', 'elevation']
    
    for d in [train_df, val_df, calib_df, test_df]:
        d[meta_cols] = d[meta_cols].fillna('unknown')

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[meta_cols])
    
    encoder_path = "outputs/meta_encoder.joblib"
    joblib.dump(encoder, encoder_path)
    print(f"Đã lưu khuôn Metadata Encoder tại: {encoder_path}")

    # 6. Lưu file
    train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
    calib_df.to_csv(os.path.join(output_dir, "calib_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    
    print(f"Hoàn tất! Số lượng mẫu: Train={len(train_df)}, Val={len(val_df)}, Calib={len(calib_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    main()