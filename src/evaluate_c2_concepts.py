
import os
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel


CONCEPT_NAMES = [
    "Pigment Network",
    "Streaks",
    "Pigmentation",
    "Regression Structures",
    "Dots & Globules",
    "Blue-whitish Veil",
    "Vascular Structures",
]


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def fmt(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]

    if len(valid) == 0:
        return "N/A"

    return f"{np.mean(valid):.4f} ± {sample_sd(valid):.4f}"


def collect_concepts(model, loader, device):
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:

            clinic = batch["clinic_img"].to(
                device, non_blocking=True
            )

            derm = batch["derm_img"].to(
                device, non_blocking=True
            )

            metadata = batch["metadata"].to(
                device, non_blocking=True
            )

            _, concept_logits = model(
                clinic,
                derm,
                meta_features=metadata,
            )

            if concept_logits is None:
                raise RuntimeError(
                    "Model does not contain concept head."
                )

            y_true.append(
                batch["concept_labels"].cpu().numpy()
            )

            y_prob.append(
                torch.sigmoid(
                    concept_logits
                ).cpu().numpy()
            )

    return (
        np.vstack(y_true),
        np.vstack(y_prob),
    )


def main():

    paths = Config.ensure_runtime_dirs()

    # =====================================================
    # Prefer local Colab image cache
    # =====================================================

    local_images = "/content/local_images"

    if os.path.isdir(local_images):
        paths["img_dir"] = local_images
        print(
            f"✅ C2 evaluation using LOCAL images: "
            f"{paths['img_dir']}"
        )
    else:
        print(
            f"⚠️ Using configured images: "
            f"{paths['img_dir']}"
        )

    # =====================================================
    # Output
    # =====================================================

    out_dir = os.path.join(
        paths["results_dir"],
        "contribution_2"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =====================================================
    # Metadata
    # =====================================================

    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    # =====================================================
    # Test dataset
    # =====================================================

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # =====================================================
    # C2 models
    # =====================================================

    experiments = [
        {
            "name": "B6_PureCBM",
            "bottleneck": "pure",
        },
        {
            "name": "Proposed_Hybrid",
            "bottleneck": "hybrid",
        },
    ]

    per_seed_rows = []
    per_concept_rows = []

    # =====================================================
    # Evaluation
    # =====================================================

    for exp in experiments:

        for seed in Config.SEEDS:

            checkpoint = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth"
            )

            if not os.path.exists(checkpoint):
                raise FileNotFoundError(
                    checkpoint
                )

            print(
                f"\nEvaluating "
                f"{exp['name']} seed={seed}"
            )

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type=exp["bottleneck"],
                use_metadata=True,
                meta_input_dim=meta_input_dim,
            ).to(device)

            model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location=device
                ),
                strict=True,
            )

            y_true, y_prob = collect_concepts(
                model,
                loader,
                device
            )

            # Fixed threshold. No Test tuning.
            y_pred = (
                y_prob >= 0.5
            ).astype(int)

            concept_f1s = []
            concept_aurocs = []

            for j, concept_name in enumerate(
                CONCEPT_NAMES
            ):

                yt = y_true[:, j]
                yp = y_pred[:, j]
                prob = y_prob[:, j]

                f1 = f1_score(
                    yt,
                    yp,
                    zero_division=0
                )

                precision = precision_score(
                    yt,
                    yp,
                    zero_division=0
                )

                recall = recall_score(
                    yt,
                    yp,
                    zero_division=0
                )

                if len(np.unique(yt)) == 2:
                    auroc = roc_auc_score(
                        yt,
                        prob
                    )
                else:
                    auroc = np.nan

                concept_f1s.append(f1)
                concept_aurocs.append(auroc)

                per_concept_rows.append({
                    "Model": exp["name"],
                    "Seed": seed,
                    "Concept": concept_name,
                    "Prevalence": float(
                        yt.mean()
                    ),
                    "AUROC": float(auroc),
                    "F1": float(f1),
                    "Precision": float(
                        precision
                    ),
                    "Recall": float(
                        recall
                    ),
                })

            macro_concept_f1 = float(
                np.mean(concept_f1s)
            )

            macro_concept_auroc = float(
                np.nanmean(concept_aurocs)
            )

            per_seed_rows.append({
                "Model": exp["name"],
                "Seed": seed,
                "Macro Concept F1":
                    macro_concept_f1,
                "Macro Concept AUROC":
                    macro_concept_auroc,
            })

            print(
                f"Macro Concept F1="
                f"{macro_concept_f1:.4f} | "
                f"Macro Concept AUROC="
                f"{macro_concept_auroc:.4f}"
            )

    # =====================================================
    # Save per-seed results
    # =====================================================

    seed_df = pd.DataFrame(
        per_seed_rows
    )

    concept_df = pd.DataFrame(
        per_concept_rows
    )

    seed_df.to_csv(
        os.path.join(
            out_dir,
            "c2_concept_seed_metrics.csv"
        ),
        index=False,
    )

    concept_df.to_csv(
        os.path.join(
            out_dir,
            "c2_concept_per_concept_metrics.csv"
        ),
        index=False,
    )

    # =====================================================
    # 3-seed summary
    # =====================================================

    summary_rows = []

    for model_name in seed_df[
        "Model"
    ].unique():

        df = seed_df[
            seed_df["Model"] == model_name
        ]

        f1_values = df[
            "Macro Concept F1"
        ].values

        auroc_values = df[
            "Macro Concept AUROC"
        ].values

        summary_rows.append({
            "Model": model_name,

            "Macro Concept F1":
                fmt(f1_values),

            "Macro Concept AUROC":
                fmt(auroc_values),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            out_dir,
            "c2_concept_summary.csv"
        ),
        index=False,
    )

    # =====================================================
    # Markdown
    # =====================================================

    md_path = os.path.join(
        out_dir,
        "c2_concept_metrics.md"
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# C2 Concept Prediction "
            "Performance\n\n"
        )

        f.write(
            "Evaluation is performed on "
            "the independent Test split. "
            "Concept predictions use a fixed "
            "0.5 sigmoid threshold. "
            "No threshold is tuned on Test.\n\n"
        )

        f.write(
            "Primary values are mean ± "
            "sample SD across seeds "
            "42, 100, and 2026.\n\n"
        )

        f.write(
            "## Overall concept performance\n\n"
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
            seed_df.to_markdown(
                index=False,
                floatfmt=".4f"
            )
        )

    print("\n=== C2 CONCEPT SUMMARY ===")
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
