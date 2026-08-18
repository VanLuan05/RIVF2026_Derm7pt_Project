
import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import (
    MultimodalDermDataset,
    calculate_dataset_weights,
    train_transforms,
    test_transforms,
)
from src.models.models import C1CrossAttentionModel
from src.train import train_model

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loaders(paths, seed):

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Train một seed cụ thể. Bỏ trống để chạy tất cả Config.SEEDS."
    )
    args = parser.parse_args()

    seeds = (
        (args.seed,)
        if args.seed is not None
        else Config.SEEDS
    )

    paths = Config.ensure_runtime_dirs()

    # C1: ưu tiên ảnh local trên Colab để tăng tốc training
    local_img_dir = "/content/local_images"

    if os.path.exists(local_img_dir):
        paths["img_dir"] = local_img_dir
        print("✅ C1 sử dụng local images:", paths["img_dir"])
    else:
        print("⚠️ Không tìm thấy local_images, dùng:", paths["img_dir"])

    required = [
        paths["train_csv"],
        paths["val_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
    ]

    missing = [
        p for p in required
        if not os.path.exists(p)
    ]

    if missing:
        raise FileNotFoundError(
            "Thiếu file:\n- " + "\n- ".join(missing)
        )

    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    disease_weights, concept_pos_weights = (
        calculate_dataset_weights(
            paths["train_csv"],
            paths["label_mapping"],
        )
    )

    results_dir = os.path.join(
        paths["results_dir"],
        "contribution_1"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    records = []

    print("=" * 70)
    print("C1 CROSS-ATTENTION TRAINING")
    print("Seeds:", list(seeds))
    print("Metadata dim:", meta_input_dim)
    print("=" * 70)

    for seed in seeds:

        set_seed(seed)

        train_loader, val_loader = make_loaders(
            paths,
            seed
        )

        model = C1CrossAttentionModel(
            num_classes=Config.NUM_CLASSES,
            meta_input_dim=meta_input_dim,
            d_model=256,
            num_heads=4,
        )

        experiment_name = (
            f"C1_CrossAttention_seed_{seed}"
        )

        print("\n" + "=" * 70)
        print("TRAIN:", experiment_name)
        print("=" * 70)

        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            disease_weights=disease_weights,
            concept_pos_weights=concept_pos_weights,
            num_epochs=Config.NUM_EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            experiment_name=experiment_name,

            # C1 không sử dụng Concept Bottleneck
            alpha=0.0,

            monitor="disease_f1",
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.MIN_DELTA,
        )

        records.append({
            "seed": seed,
            "checkpoint":
                f"{experiment_name}.pth",

            "best_epoch":
                summary["best_epoch"],

            "best_val_disease_macro_f1":
                summary[
                    "best_val_disease_macro_f1"
                ],

            "best_val_loss":
                summary["best_val_loss"],
        })

    manifest = {
        "created_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "model":
            "C1_CrossAttention",

        "fusion":
            "metadata-guided cross-attention",

        "query":
            "structured metadata",

        "key_value":
            "clinical + dermoscopic spatial tokens",

        "d_model":
            256,

        "num_heads":
            4,

        "seeds":
            list(seeds),

        "training":
            records,
    }

    if len(seeds) == 1:
        manifest_filename = (
            f"c1_training_manifest_seed_{seeds[0]}.json"
        )
    else:
        manifest_filename = "c1_training_manifest_all.json"

    manifest_path = os.path.join(
        results_dir,
        manifest_filename
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n✅ C1 training hoàn tất")
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
