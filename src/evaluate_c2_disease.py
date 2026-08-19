
import os
import json

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel
from src.run_evaluation import evaluate_single_model


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.4f} ± {sample_sd(arr):.4f}"


def main():

    paths = Config.ensure_runtime_dirs()

    # Use Colab local image cache
    if os.path.isdir("/content/local_images"):
        paths["img_dir"] = "/content/local_images"

    print("Image source:", paths["img_dir"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    workers = (
        2
        if paths["data_root"].startswith("/content/")
        else 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Test samples:", len(test_dataset))

    # Only C2 models
    experiments = [
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
    ]

    per_seed_rows = []

    for exp in experiments:

        for seed in Config.SEEDS:

            checkpoint = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth"
            )

            if not os.path.exists(checkpoint):
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint}"
                )

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality=exp["modality"],
                bottleneck_type=exp["bottleneck"],
                use_metadata=exp["meta"],
                meta_input_dim=meta_input_dim,
            ).to(device)

            state = torch.load(
                checkpoint,
                map_location=device
            )

            model.load_state_dict(
                state,
                strict=True
            )

            result = evaluate_single_model(
                model,
                test_loader,
                device,
                Config.NUM_CLASSES,
            )

            row = {
                "Model": exp["name"],
                "Seed": seed,
                "Accuracy": result["accuracy"],
                "Balanced Accuracy":
                    result["balanced_accuracy"],
                "Macro F1":
                    result["macro_f1"],
                "Macro Precision":
                    result["macro_precision"],
                "Macro Recall":
                    result["macro_recall"],
                "Macro Specificity":
                    result["macro_specificity"],
                "One-vs-Rest AUROC":
                    result["auroc"],
            }

            per_seed_rows.append(row)

            print(
                f"{exp['name']} seed={seed} | "
                f"F1={result['macro_f1']:.4f} | "
                f"BAcc={result['balanced_accuracy']:.4f} | "
                f"AUROC={result['auroc']:.4f}"
            )

    per_seed_df = pd.DataFrame(
        per_seed_rows
    )

    summary_rows = []

    metric_names = [
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Macro Precision",
        "Macro Recall",
        "Macro Specificity",
        "One-vs-Rest AUROC",
    ]

    for model_name in [
        "B6_PureCBM",
        "Proposed_Hybrid"
    ]:

        sub = per_seed_df[
            per_seed_df["Model"] == model_name
        ]

        row = {
            "Model": model_name
        }

        for metric in metric_names:
            row[metric] = mean_sd(
                sub[metric].values
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(
        summary_rows
    )

    out_dir = os.path.join(
        paths["results_dir"],
        "contribution_2"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    per_seed_csv = os.path.join(
        out_dir,
        "c2_disease_seed_metrics.csv"
    )

    summary_csv = os.path.join(
        out_dir,
        "c2_disease_summary.csv"
    )

    md_path = os.path.join(
        out_dir,
        "c2_disease_metrics.md"
    )

    per_seed_df.to_csv(
        per_seed_csv,
        index=False
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# C2 Disease Classification on Independent Test Set\n\n"
        )

        f.write(
            "Evaluation uses the corrected C2 checkpoints "
            "for B6_PureCBM and Proposed_Hybrid across "
            "the three locked seeds. The Test set is used "
            "only for final evaluation.\n\n"
        )

        f.write(
            "## Summary\n\n"
        )

        f.write(
            summary_df.to_markdown(
                index=False
            )
        )

        f.write(
            "\n\n## Per-seed results\n\n"
        )

        f.write(
            per_seed_df.to_markdown(
                index=False,
                floatfmt=".4f"
            )
        )

    print(
        "\n=== C2 DISEASE TEST SUMMARY ==="
    )

    print(
        summary_df.to_markdown(
            index=False
        )
    )

    print(
        "\nSaved:",
        md_path
    )


if __name__ == "__main__":
    main()
