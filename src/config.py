import os

# Tự động lấy thư mục gốc của dự án (Thư mục chứa thư mục src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class Config:
    # 1. Đường dẫn Dữ liệu
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    IMG_DIR = os.path.join(DATA_ROOT, "raw/images")
    TRAIN_CSV = os.path.join(DATA_ROOT, "processed/train_split.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "processed/val_split.csv")
    LABEL_MAPPING = os.path.join(DATA_ROOT, "processed/label_mapping.json")

    # 2. Đường dẫn Đầu ra (Output/Checkpoints)
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    
    # Hàm tự động sinh tên file theo tên mô hình để tránh ghi đè
    @classmethod
    def get_checkpoint_path(cls, experiment_name="best_model"):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, f"{experiment_name}.pth")