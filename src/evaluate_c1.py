
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import (
    MultimodalDermModel,
    C1CrossAttentionModel,
)
from src.run_evaluation import evaluate_single_model


def sample_sd(values):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.0

    return float(np.std(values, ddof=1))


def main():

    # ============================================================
    # 1. Runtime paths
    # ============================================================

    paths = Config.ensure_runtime_dirs()

    # Dùng ảnh local nếu Colab đã copy sẵn
    if os.path.exists("/content/local_images"):
        paths["img_dir"] = "/content/local_images"
        print("✅ Using local images:", paths["img_dir"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ============================================================
    # 2. Kiểm tra dữ liệu
    # ============================================================

    required = [
        paths["test_csv"],
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

    # ============================================================
    # 3. Test Loader
    # ============================================================

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("Test samples:", len(test_dataset))
    print("Metadata dim:", meta_input_dim)

    # ============================================================
    # 4. Output directory
    # ============================================================

    result_dir = os.path.join(
        paths["results_dir"],
        "contribution_1"
    )

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # ============================================================
    # 5. Hai mô hình cần so sánh
    #
    # B5 = concatenation baseline
    # C1 = proposed Cross-Attention
    # ============================================================

    experiments = [
        {
            "name": "B5_Dual_Metadata",
            "fusion": "Concatenation",
        },
        {
            "name": "C1_CrossAttention",
            "fusion": "Cross-Attention",
        },
    ]

    seeds = list(Config.SEEDS)

    per_seed_rows = []

    # ============================================================
    # 6. Evaluation
    # ============================================================

    for exp in experiments:

        print("\n" + "=" * 72)
        print(
            f"EVALUATING {exp['name']} "
            f"({exp['fusion']})"
        )
        print("=" * 72)

        for seed in seeds:

            checkpoint_path = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth"
            )

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Thiếu checkpoint: {checkpoint_path}"
                )

            # ----------------------------------------------------
            # B5 baseline
            # ----------------------------------------------------

            if exp["name"] == "B5_Dual_Metadata":

                model = MultimodalDermModel(
                    num_classes=Config.NUM_CLASSES,
                    num_concepts=Config.NUM_CONCEPTS,
                    modality="dual",
                    bottleneck_type="none",
                    use_metadata=True,
                    meta_input_dim=meta_input_dim,
                )

            # ----------------------------------------------------
            # C1 Cross-Attention
            # ----------------------------------------------------

            else:

                model = C1CrossAttentionModel(
                    num_classes=Config.NUM_CLASSES,
                    meta_input_dim=meta_input_dim,
                    d_model=256,
                    num_heads=4,
                )

            state = torch.load(
                checkpoint_path,
                map_location=device
            )

            model.load_state_dict(
                state,
                strict=True
            )

            model = model.to(device)

            result = evaluate_single_model(
                model,
                test_loader,
                device,
                Config.NUM_CLASSES,
            )

            row = {
                "model": exp["name"],
                "fusion": exp["fusion"],
                "seed": seed,

                "accuracy":
                    result["accuracy"],

                "balanced_accuracy":
                    result["balanced_accuracy"],

                "macro_f1":
                    result["macro_f1"],

                "macro_precision":
                    result["macro_precision"],

                "macro_recall":
                    result["macro_recall"],

                "macro_specificity":
                    result["macro_specificity"],

                "auroc":
                    result["auroc"],
            }

            per_seed_rows.append(row)

            print(
                f"Seed {seed} | "
                f"Macro-F1={result['macro_f1']:.4f} | "
                f"BAcc={result['balanced_accuracy']:.4f} | "
                f"AUROC={result['auroc']:.4f}"
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ============================================================
    # 7. Save per-seed results
    # ============================================================

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    seed_csv = os.path.join(
        result_dir,
        "c1_seed_results.csv"
    )

    per_seed_df.to_csv(
        seed_csv,
        index=False
    )

    # ============================================================
    # 8. Mean ± SD
    # ============================================================

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "macro_specificity",
        "auroc",
    ]

    summary_rows = []

    for exp in experiments:

        subset = per_seed_df[
            per_seed_df["model"]
            == exp["name"]
        ]

        row = {
            "model": exp["name"],
            "fusion": exp["fusion"],
        }

        for metric in metrics:

            values = subset[
                metric
            ].to_numpy()

            row[f"{metric}_mean"] = (
                float(np.mean(values))
            )

            row[f"{metric}_sd"] = (
                sample_sd(values)
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_csv = os.path.join(
        result_dir,
        "c1_comparison.csv"
    )

    summary_df.to_csv(
        comparison_csv,
        index=False
    )

    # ============================================================
    # 9. Markdown result
    # ============================================================

    md_path = os.path.join(
        result_dir,
        "c1_final_results.md"
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Contribution 1 - "
            "Cross-Attention Evaluation\n\n"
        )

        f.write(
            "Independent Test set evaluation "
            "across seeds 42, 100, and 2026.\n\n"
        )

        f.write(
            "| Model | Fusion | Macro-F1 | "
            "Balanced Accuracy | AUROC |\n"
        )

        f.write(
            "|---|---|---:|---:|---:|\n"
        )

        for _, row in summary_df.iterrows():

            f.write(
                f"| {row['model']} "
                f"| {row['fusion']} "
                f"| {row['macro_f1_mean']:.4f} "
                f"± {row['macro_f1_sd']:.4f} "
                f"| {row['balanced_accuracy_mean']:.4f} "
                f"± {row['balanced_accuracy_sd']:.4f} "
                f"| {row['auroc_mean']:.4f} "
                f"± {row['auroc_sd']:.4f} |\n"
            )

    print("\n✅ Evaluation files ready:")
    print(seed_csv)
    print(comparison_csv)
    print(md_path)


if __name__ == "__main__":
    main()
