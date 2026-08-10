import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Central configuration for the paper-final Derm7pt pipeline."""

    # ------------------------------------------------------------------
    # Project / data paths
    # ------------------------------------------------------------------
    BASE_DIR = str(PROJECT_ROOT)
    DATA_ROOT = os.path.join(BASE_DIR, "data")

    # Keep these under the repository root. On Colab they can be symlinked
    # to Google Drive so checkpoints/results survive a runtime reset.
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    TRAIN_CSV = os.path.join(DATA_ROOT, "processed", "train_split.csv")
    VAL_CSV = os.path.join(DATA_ROOT, "processed", "val_split.csv")
    CALIB_CSV = os.path.join(DATA_ROOT, "processed", "calib_split.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "processed", "test_split.csv")
    LABEL_MAPPING = os.path.join(DATA_ROOT, "processed", "label_mapping.json")
    IMG_DIR = os.path.join(DATA_ROOT, "raw", "images")
    META_ENCODER = os.path.join(OUTPUT_DIR, "meta_encoder.joblib")

    # Colab / Drive
    COLAB_DATASET_ROOT = "/content/drive/MyDrive/RIVF2026_Dataset"
    COLAB_DATA_ROOT = os.path.join(COLAB_DATASET_ROOT, "data")
    COLAB_LOCAL_IMAGES = "/content/local_images"

    # ------------------------------------------------------------------
    # Locked experimental protocol
    # ------------------------------------------------------------------
    NUM_CLASSES = 5
    NUM_CONCEPTS = 7
    SEEDS = (42, 100, 2026)

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 1e-4

    # Bootstrap / interpretability
    BOOTSTRAP_ITERATIONS = 1000
    BOOTSTRAP_RANDOM_STATE = 42
    ORACLE_POS_PROB = 0.95
    ORACLE_NEG_PROB = 0.05
    GRADCAM_SEED = 42

    # Final paper models. Alpha is injected at runtime for concept models.
    PAPER_EXPERIMENTS = (
        {
            "name": "B1_Clinical_Only",
            "modality": "clinic_only",
            "bottleneck": "none",
            "meta": False,
        },
        {
            "name": "B2_Derm_Only",
            "modality": "derm_only",
            "bottleneck": "none",
            "meta": False,
        },
        {
            "name": "B3_Meta_Only",
            "modality": "meta_only",
            "bottleneck": "none",
            "meta": True,
        },
        {
            "name": "B4_Dual_NoMeta",
            "modality": "dual",
            "bottleneck": "none",
            "meta": False,
        },
        {
            "name": "B5_Dual_Metadata",
            "modality": "dual",
            "bottleneck": "none",
            "meta": True,
        },
        {
            "name": "B6_PureCBM",
            "modality": "dual",
            "bottleneck": "pure",
            "meta": True,
        },
        {
            "name": "Proposed_Hybrid",
            "modality": "dual",
            "bottleneck": "hybrid",
            "meta": True,
        },
    )

    @classmethod
    def is_colab(cls) -> bool:
        return os.path.isdir("/content")

    @classmethod
    def runtime_paths(cls) -> dict:
        """
        Resolve paths consistently for local machines and Google Colab.

        Colab policy:
        - CSV/meta are read from Google Drive when available.
        - Images prefer /content/local_images for faster I/O.
        - outputs/results remain under the repository root; the notebook
          symlinks them to Drive.
        """
        data_root = (
            cls.COLAB_DATA_ROOT
            if os.path.isdir(cls.COLAB_DATA_ROOT)
            else cls.DATA_ROOT
        )

        img_dir = (
            cls.COLAB_LOCAL_IMAGES
            if os.path.isdir(cls.COLAB_LOCAL_IMAGES)
            else os.path.join(data_root, "raw", "images")
        )

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
            "meta_encoder": os.path.join(cls.OUTPUT_DIR, "meta_encoder.joblib"),
            "selected_alpha": os.path.join(cls.OUTPUT_DIR, "selected_alpha.json"),
            "output_dir": cls.OUTPUT_DIR,
            "results_dir": cls.RESULTS_DIR,
        }

    @classmethod
    def ensure_runtime_dirs(cls) -> dict:
        paths = cls.runtime_paths()
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["results_dir"], exist_ok=True)
        return paths

    @classmethod
    def get_checkpoint_path(cls, experiment_name: str = "best_model") -> str:
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, f"{experiment_name}.pth")

    @classmethod
    def load_selected_alpha(cls) -> float:
        """
        Read alpha selected by run_alpha_ablation.py.
        No silent fallback is allowed in the paper-final protocol.
        """
        path = cls.runtime_paths()["selected_alpha"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy {path}.\n"
                "Hãy chạy trước: python -m src.run_alpha_ablation"
            )

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if "selected_alpha" not in payload:
            raise KeyError(
                f"{path} không có khóa 'selected_alpha'. "
                "Hãy chạy lại run_alpha_ablation.py."
            )

        alpha = float(payload["selected_alpha"])
        if alpha < 0:
            raise ValueError("selected_alpha phải >= 0.")
        return alpha

    @classmethod
    def print_runtime_paths(cls) -> None:
        paths = cls.runtime_paths()
        print("=" * 72)
        print("RIVF2026 RUNTIME PATHS")
        print("=" * 72)
        for key in (
            "project_root",
            "data_root",
            "img_dir",
            "train_csv",
            "val_csv",
            "calib_csv",
            "test_csv",
            "label_mapping",
            "meta_encoder",
            "selected_alpha",
            "output_dir",
            "results_dir",
        ):
            print(f"{key:15s}: {paths[key]}")
        print("=" * 72)

        if cls.is_colab():
            if os.path.isdir(cls.COLAB_LOCAL_IMAGES):
                print("[OK] Images are read from /content/local_images")
            else:
                print(
                    "[INFO] /content/local_images not found; "
                    "images fall back to data/raw/images."
                )


if __name__ == "__main__":
    Config.print_runtime_paths()