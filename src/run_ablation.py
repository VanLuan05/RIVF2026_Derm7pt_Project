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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loaders(paths: dict, seed: int):
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


def _remove_stale_run_files(checkpoint_name: str) -> None:
    """
    Prevent a failed new run from being confused with an old checkpoint.
    """
    checkpoint_path = Config.get_checkpoint_path(checkpoint_name)
    metadata_path = os.path.splitext(checkpoint_path)[0] + "_training.json"

    for path in (checkpoint_path, metadata_path):
        if os.path.exists(path):
            os.remove(path)


def run_experiment(
    exp: dict,
    seed: int,
    paths: dict,
    meta_input_dim: int,
    disease_weights,
    concept_pos_weights,
    selected_alpha: float,
):
    set_seed(seed)
    train_loader, val_loader = make_loaders(paths, seed)

    alpha = (
        selected_alpha
        if exp["bottleneck"] in {"pure", "hybrid", "multitask"}
        else 0.0
    )

    checkpoint_name = f"{exp['name']}_seed_{seed}"
    _remove_stale_run_files(checkpoint_name)

    print("\n" + "=" * 78)
    print(
        f"TRAIN {checkpoint_name} | "
        f"modality={exp['modality']} | "
        f"bottleneck={exp['bottleneck']} | "
        f"metadata={exp['meta']} | alpha={alpha}"
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
        alpha=alpha,
        monitor="disease_f1",
        patience=Config.EARLY_STOPPING_PATIENCE,
        min_delta=Config.MIN_DELTA,
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
        "alpha": float(alpha),
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


def main():
    paths = Config.ensure_runtime_dirs()

    required_files = [
        paths["train_csv"],
        paths["val_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
        paths["selected_alpha"],
    ]
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file cần thiết:\n- "
            + "\n- ".join(missing)
            + "\nHãy chạy prepare_data, check_distribution và "
              "run_alpha_ablation trước."
        )

    selected_alpha = Config.load_selected_alpha()

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

    disease_weights, concept_pos_weights = calculate_dataset_weights(
        paths["train_csv"],
        paths["label_mapping"],
    )

    experiments = list(Config.PAPER_EXPERIMENTS)
    expected_runs = len(experiments) * len(Config.SEEDS)
    if expected_runs != 21:
        raise RuntimeError(
            f"Cấu hình hiện tạo {expected_runs} models, không phải 21."
        )

    print("\n" + "#" * 78)
    print("PAPER-FINAL ABLATION PROTOCOL")
    print(f"Selected alpha      : {selected_alpha}")
    print(f"Seeds               : {list(Config.SEEDS)}")
    print(f"Metadata dimension  : {meta_input_dim}")
    print(f"Max epochs          : {Config.NUM_EPOCHS}")
    print(f"Learning rate       : {Config.LEARNING_RATE}")
    print(f"Weight decay        : {Config.WEIGHT_DECAY}")
    print(f"Early-stop patience : {Config.EARLY_STOPPING_PATIENCE}")
    print("Checkpoint monitor  : Validation Disease Macro-F1")
    print("Expected             : 7 architectures x 3 seeds = 21")
    print("#" * 78)

    training_records = []

    for exp_idx, exp in enumerate(experiments, start=1):
        print("\n" + "*" * 78)
        print(
            f"ARCHITECTURE {exp_idx}/{len(experiments)}: "
            f"{exp['name']}"
        )
        print("*" * 78)

        for seed in Config.SEEDS:
            training_records.append(
                run_experiment(
                    exp=exp,
                    seed=seed,
                    paths=paths,
                    meta_input_dim=meta_input_dim,
                    disease_weights=disease_weights,
                    concept_pos_weights=concept_pos_weights,
                    selected_alpha=selected_alpha,
                )
            )

    if len(training_records) != 21:
        raise RuntimeError(
            f"Training kết thúc với {len(training_records)}/21 runs."
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_alpha": selected_alpha,
        "seeds": list(Config.SEEDS),
        "num_architectures": len(experiments),
        "num_completed_models": len(training_records),
        "optimizer": "AdamW",
        "learning_rate": Config.LEARNING_RATE,
        "weight_decay": Config.WEIGHT_DECAY,
        "max_epochs": Config.NUM_EPOCHS,
        "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
        "checkpoint_monitor": "validation_disease_macro_f1",
        "metadata_input_dim": meta_input_dim,
        "experiments": training_records,
    }

    manifest_path = os.path.join(
        paths["results_dir"],
        "ablation_training_manifest.json",
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("HOÀN TẤT PAPER ABLATION")
    print("Đã train đủ 21 checkpoints.")
    print(f"Manifest: {manifest_path}")
    print("Tiếp theo: python -m src.run_evaluation")
    print("=" * 78)


if __name__ == "__main__":
    main()