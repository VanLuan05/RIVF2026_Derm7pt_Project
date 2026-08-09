import json
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
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


def sample_sd(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


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
    generator = torch.Generator().manual_seed(seed)
    workers = 2 if paths["data_root"].startswith("/content/") else 0
    return (
        DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=workers,
            generator=generator,
        ),
        DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=workers,
        ),
    )


def main():
    paths = Config.runtime_paths()
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)

    if not os.path.exists(paths["meta_encoder"]):
        raise FileNotFoundError("Thiếu meta_encoder.joblib. Hãy chạy prepare_data.py trước.")

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())
    disease_weights, concept_pos_weights = calculate_dataset_weights(
        paths["train_csv"], paths["label_mapping"]
    )

    # Final confirmation sweep. Không dùng Test để chọn alpha.
    candidate_alphas = [2.0, 3.0]
    rows = []
    raw = {}

    for alpha in candidate_alphas:
        disease_scores = []
        concept_scores = []
        raw[str(alpha)] = {}

        print("\n" + "=" * 72)
        print(f"ALPHA CONFIRMATION: alpha={alpha}")
        print("=" * 72)

        for seed in Config.SEEDS:
            set_seed(seed)
            train_loader, val_loader = make_loaders(paths, seed)
            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type="hybrid",
                use_metadata=True,
                meta_input_dim=meta_input_dim,
            )

            tmp_name = f"alpha_tuning_{str(alpha).replace('.', 'p')}_seed_{seed}"
            summary = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                disease_weights=disease_weights,
                concept_pos_weights=concept_pos_weights,
                num_epochs=Config.NUM_EPOCHS,
                learning_rate=Config.LEARNING_RATE,
                weight_decay=Config.WEIGHT_DECAY,
                experiment_name=tmp_name,
                alpha=alpha,
                monitor="disease_f1",
                patience=Config.EARLY_STOPPING_PATIENCE,
            )

            d_f1 = float(summary["best_val_disease_macro_f1"])
            c_f1 = summary["val_concept_macro_f1_at_best_epoch"]
            c_f1 = np.nan if c_f1 is None else float(c_f1)
            disease_scores.append(d_f1)
            concept_scores.append(c_f1)
            raw[str(alpha)][str(seed)] = summary

        rows.append(
            {
                "Alpha": alpha,
                "Seed 42 Disease F1": disease_scores[0],
                "Seed 100 Disease F1": disease_scores[1],
                "Seed 2026 Disease F1": disease_scores[2],
                "Mean Disease F1": float(np.mean(disease_scores)),
                "SD Disease F1": sample_sd(disease_scores),
                "Mean Concept F1": float(np.nanmean(concept_scores)),
                "SD Concept F1": sample_sd([x for x in concept_scores if not np.isnan(x)]),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["Mean Disease F1", "Mean Concept F1"], ascending=False
    )
    selected_alpha = float(df.iloc[0]["Alpha"])

    md_path = os.path.join(paths["results_dir"], "alpha_ablation_final.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Alpha Ablation on Validation Set\n\n")
        f.write(
            "Alpha is selected using mean Validation Disease Macro-F1 across three seeds. "
            "Concept Macro-F1 is used as a secondary criterion. The Test set is not used for selection.\n\n"
        )
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write(f"\n\n**Selected alpha: {selected_alpha:.1f}**\n")

    selection = {
        "selected_alpha": selected_alpha,
        "selection_metric": "mean_validation_disease_macro_f1_across_3_seeds",
        "seeds": list(Config.SEEDS),
        "candidate_alphas": candidate_alphas,
        "protocol": {
            "optimizer": "AdamW",
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "weighted_disease_loss": True,
            "weighted_concept_loss": True,
            "checkpoint_monitor": "validation_disease_macro_f1",
            "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
        },
        "raw_training_summaries": raw,
    }
    with open(os.path.join(paths["output_dir"], "selected_alpha.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nSelected alpha = {selected_alpha:.1f}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()