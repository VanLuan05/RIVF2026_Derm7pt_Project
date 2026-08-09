import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Config:
    """Cấu hình tập trung cho pipeline thí nghiệm paper-ready."""

    # Project paths (local defaults)
    BASE_DIR = BASE_DIR
    DATA_ROOT = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    TRAIN_CSV = os.path.join(DATA_ROOT, "processed", "train_split.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "processed", "val_split.csv")
    CALIB_CSV = os.path.join(DATA_ROOT, "processed", "calib_split.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "processed", "test_split.csv")
    LABEL_MAPPING = os.path.join(DATA_ROOT, "processed", "label_mapping.json")
    IMG_DIR = os.path.join(DATA_ROOT, "raw", "images")
    META_ENCODER = os.path.join(OUTPUT_DIR, "meta_encoder.joblib")

    # Experiment protocol
    NUM_CLASSES = 5
    NUM_CONCEPTS = 7
    SEEDS = (42, 100, 2026)
    FINAL_ALPHA = 3.0  # Fallback; run_alpha_ablation.py có thể ghi đè qua selected_alpha.json
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    @classmethod
    def get_checkpoint_path(cls, experiment_name="best_model"):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, f"{experiment_name}.pth")

    @classmethod
    def runtime_paths(cls):
        """
        Trả về đường dẫn phù hợp cho Local hoặc Google Colab.
        Không thay đổi logic dữ liệu; chỉ chuẩn hóa cách các script tìm file.
        """
        colab_data_root = "/content/drive/MyDrive/RIVF2026_Dataset/data"
        if os.path.exists(colab_data_root):
            data_root = colab_data_root
        else:
            data_root = cls.DATA_ROOT

        local_colab_images = "/content/local_images"
        if os.path.exists(local_colab_images):
            img_dir = local_colab_images
        else:
            img_dir = os.path.join(data_root, "raw", "images")

        return {
            "data_root": data_root,
            "img_dir": img_dir,
            "train_csv": os.path.join(data_root, "processed", "train_split.csv"),
            "val_csv": os.path.join(data_root, "processed", "val_split.csv"),
            "calib_csv": os.path.join(data_root, "processed", "calib_split.csv"),
            "test_csv": os.path.join(data_root, "processed", "test_split.csv"),
            "label_mapping": os.path.join(data_root, "processed", "label_mapping.json"),
            "meta_encoder": cls.META_ENCODER,
            "output_dir": cls.OUTPUT_DIR,
            "results_dir": cls.RESULTS_DIR,
        }