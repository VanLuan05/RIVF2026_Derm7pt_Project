import os
from pathlib import Path


# =========================================================
# Project root
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """
    Cấu hình tập trung cho pipeline RIVF2026 Derm7pt.

    Thiết kế đường dẫn cuối cùng:
    - Metadata / CSV nguồn: ưu tiên Google Drive khi chạy Colab.
    - Ảnh: ưu tiên /content/local_images để tăng tốc I/O trên Colab.
    - outputs/ và results/: nằm trong thư mục project. Khi Colab tạo symlink
      hai thư mục này sang Google Drive, mọi checkpoint/kết quả sẽ tự động
      được lưu bền vững trên Drive.

    run_ablation.py sử dụng Config.runtime_paths() và train 7 kiến trúc x
    3 seeds = 21 checkpoints sau khi alpha đã được khóa bằng Validation.
    """

    # =====================================================
    # Project paths - local/default
    # =====================================================
    BASE_DIR = str(PROJECT_ROOT)
    DATA_ROOT = os.path.join(BASE_DIR, "data")

    # Hai đường dẫn này cố ý nằm trong project.
    # Trên Colab, notebook sẽ symlink chúng sang Google Drive:
    #   outputs -> /content/drive/MyDrive/RIVF2026_Dataset/outputs
    #   results -> /content/drive/MyDrive/RIVF2026_Dataset/results
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # Local/default dataset paths
    TRAIN_CSV = os.path.join(DATA_ROOT, "processed", "train_split.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "processed", "val_split.csv")
    CALIB_CSV = os.path.join(DATA_ROOT, "processed", "calib_split.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "processed", "test_split.csv")
    LABEL_MAPPING = os.path.join(DATA_ROOT, "processed", "label_mapping.json")
    IMG_DIR = os.path.join(DATA_ROOT, "raw", "images")
    META_ENCODER = os.path.join(OUTPUT_DIR, "meta_encoder.joblib")

    # =====================================================
    # Colab / Google Drive paths
    # =====================================================
    COLAB_DATASET_ROOT = "/content/drive/MyDrive/RIVF2026_Dataset"
    COLAB_DATA_ROOT = os.path.join(COLAB_DATASET_ROOT, "data")
    COLAB_LOCAL_IMAGES = "/content/local_images"

    # =====================================================
    # Experiment protocol - LOCKED settings
    # =====================================================
    NUM_CLASSES = 5
    NUM_CONCEPTS = 7

    # Independent training seeds used throughout the paper.
    SEEDS = (42, 100, 2026)

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    # Không đặt FINAL_ALPHA ở đây.
    # Alpha cuối cùng phải được tạo bởi:
    #   python -m src.run_alpha_ablation
    # và lưu tại:
    #   outputs/selected_alpha.json
    # run_ablation.py sẽ bắt buộc đọc file đó trước khi train 21 models.

    # =====================================================
    # Runtime helpers
    # =====================================================
    @classmethod
    def is_colab(cls) -> bool:
        """Nhận biết môi trường Google Colab theo cấu trúc /content."""
        return os.path.isdir("/content")

    @classmethod
    def get_checkpoint_path(cls, experiment_name: str = "best_model") -> str:
        """
        Trả về đường dẫn checkpoint thống nhất.

        train.py gọi hàm này. OUTPUT_DIR có thể là symlink sang Google Drive,
        nên checkpoint vẫn được lưu bền vững trong workflow Colab hiện tại.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, f"{experiment_name}.pth")

    @classmethod
    def runtime_paths(cls) -> dict:
        """
        Trả về toàn bộ đường dẫn runtime dùng chung cho các script.

        Quy tắc:
        1. Nếu Google Drive dataset tồn tại -> CSV/meta đọc từ Drive.
        2. Nếu /content/local_images tồn tại -> ảnh đọc từ local Colab.
        3. Nếu không có local_images -> ảnh fallback về data/raw/images.
        4. outputs/results luôn dùng thư mục project; trên Colab có thể symlink
           hai thư mục này sang Drive để tránh mất checkpoint khi runtime reset.
        """

        # -------------------------------------------------
        # Data root
        # -------------------------------------------------
        if os.path.isdir(cls.COLAB_DATA_ROOT):
            data_root = cls.COLAB_DATA_ROOT
        else:
            data_root = cls.DATA_ROOT

        # -------------------------------------------------
        # Image directory
        # -------------------------------------------------
        if os.path.isdir(cls.COLAB_LOCAL_IMAGES):
            img_dir = cls.COLAB_LOCAL_IMAGES
        else:
            img_dir = os.path.join(data_root, "raw", "images")

        # -------------------------------------------------
        # Output/result paths
        # -------------------------------------------------
        # Giữ paths dưới project để tương thích trực tiếp với symlink:
        #   project/outputs -> Drive/outputs
        #   project/results -> Drive/results
        output_dir = cls.OUTPUT_DIR
        results_dir = cls.RESULTS_DIR

        return {
            "project_root": cls.BASE_DIR,
            "data_root": data_root,
            "img_dir": img_dir,
            "train_csv": os.path.join(data_root, "processed", "train_split.csv"),
            "val_csv": os.path.join(data_root, "processed", "val_split.csv"),
            "calib_csv": os.path.join(data_root, "processed", "calib_split.csv"),
            "test_csv": os.path.join(data_root, "processed", "test_split.csv"),
            "label_mapping": os.path.join(
                data_root, "processed", "label_mapping.json"
            ),
            "meta_encoder": os.path.join(output_dir, "meta_encoder.joblib"),
            "output_dir": output_dir,
            "results_dir": results_dir,
        }

    @classmethod
    def ensure_runtime_dirs(cls) -> dict:
        """Tạo outputs/results nếu chưa tồn tại và trả lại runtime paths."""
        paths = cls.runtime_paths()

        os.makedirs(
            paths["output_dir"],
            exist_ok=True,
        )

        os.makedirs(
            paths["results_dir"],
            exist_ok=True,
        )

        return paths

    @classmethod
    def print_runtime_paths(cls) -> None:
        """In đường dẫn thực tế để kiểm tra trước khi bắt đầu training."""
        paths = cls.runtime_paths()

        print("=" * 72)
        print("RIVF2026 RUNTIME PATHS")
        print("=" * 72)

        print(
            f"Project root : "
            f"{paths['project_root']}"
        )

        print(
            f"Data root    : "
            f"{paths['data_root']}"
        )

        print(
            f"Image dir    : "
            f"{paths['img_dir']}"
        )

        print(
            f"Train CSV    : "
            f"{paths['train_csv']}"
        )

        print(
            f"Val CSV      : "
            f"{paths['val_csv']}"
        )

        print(
            f"Calib CSV    : "
            f"{paths['calib_csv']}"
        )

        print(
            f"Test CSV     : "
            f"{paths['test_csv']}"
        )

        print(
            f"Label map    : "
            f"{paths['label_mapping']}"
        )

        print(
            f"Meta encoder : "
            f"{paths['meta_encoder']}"
        )

        print(
            f"Output dir   : "
            f"{paths['output_dir']}"
        )

        print(
            f"Results dir  : "
            f"{paths['results_dir']}"
        )

        print("=" * 72)

        if cls.is_colab():

            if os.path.isdir(
                cls.COLAB_LOCAL_IMAGES
            ):

                print(
                    "[OK] Colab đang đọc ảnh từ "
                    "/content/local_images"
                )

            else:

                print(
                    "[INFO] Không thấy /content/local_images; "
                    "ảnh sẽ được đọc từ data/raw/images."
                )


if __name__ == "__main__":
    Config.print_runtime_paths()