import json
import os
import random
import warnings
from datetime import datetime

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import (
    MultimodalDermDataset,
    calculate_dataset_weights,
    test_transforms,
    train_transforms,
)
from src.models.models import MultimodalDermModel
from src.train import train_model

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int) -> None:
    """Cố định random seed cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# Data loaders
# =========================================================
def make_loaders(paths: dict, seed: int):
    """Tạo Train/Validation loaders với đúng encoder và transforms đã khóa."""
    train_dataset = MultimodalDermDataset(
        paths["train_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=train_transforms,
    )

    val_dataset = MultimodalDermDataset(
        paths["val_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    workers = 2 if paths["data_root"].startswith("/content/") else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        generator=generator,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


# =========================================================
# Locked alpha
# =========================================================
def load_selected_alpha(output_dir: str) -> float:
    """
    Chỉ cho phép train paper experiments sau khi alpha đã được chọn trên Validation.

    run_alpha_ablation.py phải sinh:
        outputs/selected_alpha.json

    Không fallback âm thầm về một alpha hard-code để tránh vô tình train Proposed
    trước khi hyperparameter được khóa.
    """
    selection_path = os.path.join(output_dir, "selected_alpha.json")

    if not os.path.exists(selection_path):
        raise FileNotFoundError(
            "Không tìm thấy outputs/selected_alpha.json.\n"
            "Hãy chạy trước: python -m src.run_alpha_ablation\n"
            "Sau khi alpha được chọn chỉ bằng Validation, mới chạy run_ablation.py."
        )

    with open(selection_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "selected_alpha" not in payload:
        raise KeyError(
            f"File {selection_path} không có khóa 'selected_alpha'. "
            "Hãy chạy lại run_alpha_ablation.py."
        )

    selected_alpha = float(payload["selected_alpha"])

    if selected_alpha < 0:
        raise ValueError("selected_alpha phải >= 0.")

    return selected_alpha


# =========================================================
# Single experiment
# =========================================================
def run_experiment(
    exp: dict,
    seed: int,
    paths: dict,
    meta_input_dim: int,
    disease_weights,
    concept_pos_weights,
):
    """Train một architecture tại một seed và lưu best checkpoint theo Val Disease Macro-F1."""
    set_seed(seed)
    train_loader, val_loader = make_loaders(paths, seed)

    checkpoint_name = f"{exp['name']}_seed_{seed}"

    print("\n" + "=" * 78)
    print(
        f"TRAIN {checkpoint_name} | "
        f"modality={exp['modality']} | "
        f"bottleneck={exp['bottleneck']} | "
        f"metadata={exp['meta']} | alpha={exp['alpha']}"
    )
    print("=" * 78)

    model = MultimodalDermModel(
        num_classes=Config.NUM_CLASSES,
        num_concepts=Config.NUM_CONCEPTS,
        modality=exp["modality"],
        bottleneck_type=exp["bottleneck"],
        use_metadata=exp["meta"],
        meta_input_dim=meta_input_dim,
    )

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        disease_weights=disease_weights,
        concept_pos_weights=concept_pos_weights,
        num_epochs=Config.NUM_EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        experiment_name=checkpoint_name,
        alpha=exp["alpha"],
        monitor="disease_f1",
        patience=Config.EARLY_STOPPING_PATIENCE,
    )

    print(
        f"[DONE] {checkpoint_name} | "
        f"Best epoch={summary['best_epoch']} | "
        f"Best Val Disease Macro-F1="
        f"{summary['best_val_disease_macro_f1']:.4f}"
    )

    return {
        "experiment": exp["name"],
        "seed": int(seed),
        "modality": exp["modality"],
        "bottleneck": exp["bottleneck"],
        "use_metadata": bool(exp["meta"]),
        "alpha": float(exp["alpha"]),
        "checkpoint": f"{checkpoint_name}.pth",
        "best_epoch": int(summary["best_epoch"]),
        "best_val_disease_macro_f1": float(
            summary["best_val_disease_macro_f1"]
        ),
        "best_val_loss": float(summary["best_val_loss"]),
        "val_concept_macro_f1_at_best_epoch": summary.get(
            "val_concept_macro_f1_at_best_epoch"
        ),
    }


# =========================================================
# Main paper ablation: 7 architectures x 3 seeds = 21 models
# =========================================================
def main():
    paths = Config.runtime_paths()

    os.makedirs(paths["output_dir"], exist_ok=True)
    os.makedirs(paths["results_dir"], exist_ok=True)

    required_files = [
        paths["train_csv"],
        paths["val_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
    ]

    missing = [
        path
        for path in required_files
        if not os.path.exists(path)
    ]

    if missing:
        raise FileNotFoundError(
            "Thiếu file cần thiết:\n- "
            + "\n- ".join(missing)
            + "\nHãy chạy trước: python -m src.data.prepare_data"
        )

    # =====================================================
    # Metadata dimension
    # =====================================================
    # Encoder được fit chỉ trên Train trong prepare_data.py.
    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    # =====================================================
    # Training weights
    # =====================================================
    # Chỉ tính từ Train split.
    disease_weights, concept_pos_weights = (
        calculate_dataset_weights(
            paths["train_csv"],
            paths["label_mapping"],
        )
    )

    # =====================================================
    # Alpha đã được chọn trên Validation
    # =====================================================
    selected_alpha = load_selected_alpha(
        paths["output_dir"]
    )

    print("\n" + "#" * 78)
    print("PAPER ABLATION PROTOCOL")

    print(
        f"Seeds              : "
        f"{list(Config.SEEDS)}"
    )

    print(
        f"Selected alpha     : "
        f"{selected_alpha}"
    )

    print(
        f"Metadata dimension : "
        f"{meta_input_dim}"
    )

    print(
        f"Epochs max         : "
        f"{Config.NUM_EPOCHS}"
    )

    print(
        f"Learning rate      : "
        f"{Config.LEARNING_RATE}"
    )

    print(
        f"Weight decay       : "
        f"{Config.WEIGHT_DECAY}"
    )

    print(
        f"Early-stop patience: "
        f"{Config.EARLY_STOPPING_PATIENCE}"
    )

    print(
        "Checkpoint monitor : "
        "Validation Disease Macro-F1"
    )

    print(
        "Expected models    : "
        "7 architectures x 3 seeds = 21 checkpoints"
    )

    print("#" * 78)

    # =====================================================
    # FINAL PAPER ARCHITECTURES
    # =====================================================
    experiments = [

        # -------------------------------------------------
        # B1 — Clinical image only
        # -------------------------------------------------
        {
            "name": "B1_Clinical_Only",
            "modality": "clinic_only",
            "bottleneck": "none",
            "meta": False,
            "alpha": 0.0,
        },

        # -------------------------------------------------
        # B2 — Dermoscopy only
        # -------------------------------------------------
        {
            "name": "B2_Derm_Only",
            "modality": "derm_only",
            "bottleneck": "none",
            "meta": False,
            "alpha": 0.0,
        },

        # -------------------------------------------------
        # B3 — Metadata only
        # -------------------------------------------------
        {
            "name": "B3_Meta_Only",
            "modality": "meta_only",
            "bottleneck": "none",
            "meta": True,
            "alpha": 0.0,
        },

        # -------------------------------------------------
        # B4 — Clinical + Dermoscopy
        # No metadata
        # No concept bottleneck
        # -------------------------------------------------
        {
            "name": "B4_Dual_NoMeta",
            "modality": "dual",
            "bottleneck": "none",
            "meta": False,
            "alpha": 0.0,
        },

        # -------------------------------------------------
        # B5 — Clinical + Dermoscopy + Metadata
        # No concepts
        # -------------------------------------------------
        {
            "name": "B5_Dual_Metadata",
            "modality": "dual",
            "bottleneck": "none",
            "meta": True,
            "alpha": 0.0,
        },

        # -------------------------------------------------
        # B6 — Pure Concept Bottleneck Model
        # Disease prediction phụ thuộc concepts
        # -------------------------------------------------
        {
            "name": "B6_PureCBM",
            "modality": "dual",
            "bottleneck": "pure",
            "meta": True,
            "alpha": selected_alpha,
        },

        # -------------------------------------------------
        # Proposed Hybrid Concept Bottleneck Model
        #
        # Multimodal features
        #       +
        # predicted clinical concepts
        #       ↓
        # Disease classifier
        # -------------------------------------------------
        {
            "name": "Proposed_Hybrid",
            "modality": "dual",
            "bottleneck": "hybrid",
            "meta": True,
            "alpha": selected_alpha,
        },
    ]

    # =====================================================
    # Safety check
    # =====================================================
    expected_runs = (
        len(experiments)
        * len(Config.SEEDS)
    )

    if expected_runs != 21:
        raise RuntimeError(
            f"Cấu hình hiện tạo {expected_runs} models, "
            "không phải 21."
        )

    training_records = []

    # =====================================================
    # Train 7 architectures × 3 seeds
    # =====================================================
    for exp_idx, exp in enumerate(
        experiments,
        start=1,
    ):

        print("\n" + "*" * 78)

        print(
            f"ARCHITECTURE "
            f"{exp_idx}/{len(experiments)}: "
            f"{exp['name']}"
        )

        print("*" * 78)

        for seed in Config.SEEDS:

            record = run_experiment(
                exp=exp,
                seed=seed,
                paths=paths,
                meta_input_dim=meta_input_dim,
                disease_weights=disease_weights,
                concept_pos_weights=concept_pos_weights,
            )

            training_records.append(
                record
            )

    # =====================================================
    # Training manifest
    # =====================================================
    manifest = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "selected_alpha": selected_alpha,

        "seeds": list(
            Config.SEEDS
        ),

        "num_architectures": len(
            experiments
        ),

        "num_expected_models": expected_runs,

        "num_completed_models": len(
            training_records
        ),

        "checkpoint_monitor":
            "validation_disease_macro_f1",

        "optimizer": "AdamW",

        "learning_rate":
            Config.LEARNING_RATE,

        "weight_decay":
            Config.WEIGHT_DECAY,

        "max_epochs":
            Config.NUM_EPOCHS,

        "early_stopping_patience":
            Config.EARLY_STOPPING_PATIENCE,

        "metadata_input_dim":
            meta_input_dim,

        "experiments":
            training_records,
    }

    manifest_path = os.path.join(
        paths["results_dir"],
        "ablation_training_manifest.json",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # =====================================================
    # Done
    # =====================================================
    print("\n" + "=" * 78)

    print(
        "HOÀN TẤT PAPER ABLATION"
    )

    print(
        f"Đã train: "
        f"{len(training_records)}/21 checkpoints"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        "\nTiếp theo chạy:"
    )

    print(
        "  python -m src.run_evaluation"
    )

    print(
        "  python -m src.concept_evaluation"
    )

    print(
        "  python -m src.run_intervention"
    )

    print(
        "  python -m src.sequential_cbm"
    )

    print(
        "  python -m src.bootstrap_eval"
    )

    print(
        "  python -m src.gradcam_vis"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()