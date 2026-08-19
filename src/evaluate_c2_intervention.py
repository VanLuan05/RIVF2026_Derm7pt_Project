
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import (
    MultimodalDermDataset,
    test_transforms,
)
from src.models.models import MultimodalDermModel


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.4f} ± {sample_sd(arr):.4f}"


def soft_oracle(concepts):
    return torch.where(
        concepts > 0.5,
        torch.full_like(
            concepts,
            Config.ORACLE_POS_PROB
        ),
        torch.full_like(
            concepts,
            Config.ORACLE_NEG_PROB
        ),
    )


def evaluate_intervention(
    model,
    loader,
    device,
):

    model.eval()

    labels_all = []
    pred_ai_all = []
    pred_oracle_all = []

    with torch.inference_mode():

        for batch in loader:

            clinic = batch[
                "clinic_img"
            ].to(
                device,
                non_blocking=True
            )

            derm = batch[
                "derm_img"
            ].to(
                device,
                non_blocking=True
            )

            metadata = batch[
                "metadata"
            ].to(
                device,
                non_blocking=True
            )

            labels = batch[
                "label_disease"
            ].cpu().numpy()

            gt_concepts = batch[
                "concept_labels"
            ].to(
                device,
                non_blocking=True
            ).float()

            oracle_probs = soft_oracle(
                gt_concepts
            )

            # ---------------------------------------------
            # Normal AI prediction
            # ---------------------------------------------

            logits_ai, _ = model(
                clinic,
                derm,
                meta_features=metadata,
            )

            # ---------------------------------------------
            # Ground-truth concept intervention
            # ---------------------------------------------

            logits_oracle, _ = model(
                clinic,
                derm,
                meta_features=metadata,
                intervention_probs=oracle_probs,
            )

            pred_ai = torch.argmax(
                logits_ai,
                dim=1
            ).cpu().numpy()

            pred_oracle = torch.argmax(
                logits_oracle,
                dim=1
            ).cpu().numpy()

            labels_all.extend(labels)
            pred_ai_all.extend(pred_ai)
            pred_oracle_all.extend(
                pred_oracle
            )

    classes = list(
        range(Config.NUM_CLASSES)
    )

    results = {
        "accuracy_ai":
            accuracy_score(
                labels_all,
                pred_ai_all
            ),

        "accuracy_oracle":
            accuracy_score(
                labels_all,
                pred_oracle_all
            ),

        "balanced_accuracy_ai":
            balanced_accuracy_score(
                labels_all,
                pred_ai_all
            ),

        "balanced_accuracy_oracle":
            balanced_accuracy_score(
                labels_all,
                pred_oracle_all
            ),

        "macro_f1_ai":
            f1_score(
                labels_all,
                pred_ai_all,
                labels=classes,
                average="macro",
                zero_division=0,
            ),

        "macro_f1_oracle":
            f1_score(
                labels_all,
                pred_oracle_all,
                labels=classes,
                average="macro",
                zero_division=0,
            ),
    }

    return {
        k: float(v)
        for k, v in results.items()
    }


def main():

    paths = Config.ensure_runtime_dirs()

    # =====================================================
    # Prefer local image cache
    # =====================================================

    if os.path.isdir(
        "/content/local_images"
    ):
        paths["img_dir"] = (
            "/content/local_images"
        )

    print(
        "Image source:",
        paths["img_dir"]
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
    # Test loader
    # =====================================================

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=
            paths["meta_encoder"],
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=
            torch.cuda.is_available(),
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # =====================================================
    # Models
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

    for exp in experiments:

        for seed in Config.SEEDS:

            checkpoint = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth"
            )

            if not os.path.exists(
                checkpoint
            ):
                raise FileNotFoundError(
                    checkpoint
                )

            model = MultimodalDermModel(
                num_classes=
                    Config.NUM_CLASSES,
                num_concepts=
                    Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type=
                    exp["bottleneck"],
                use_metadata=True,
                meta_input_dim=
                    meta_input_dim,
            ).to(device)

            model.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location=device
                ),
                strict=True,
            )

            result = evaluate_intervention(
                model,
                test_loader,
                device,
            )

            row = {
                "Model": exp["name"],
                "Seed": seed,

                "Accuracy AI":
                    result["accuracy_ai"],

                "Accuracy Oracle":
                    result[
                        "accuracy_oracle"
                    ],

                "Delta Accuracy":
                    result[
                        "accuracy_oracle"
                    ]
                    -
                    result[
                        "accuracy_ai"
                    ],

                "Balanced Accuracy AI":
                    result[
                        "balanced_accuracy_ai"
                    ],

                "Balanced Accuracy Oracle":
                    result[
                        "balanced_accuracy_oracle"
                    ],

                "Delta Balanced Accuracy":
                    result[
                        "balanced_accuracy_oracle"
                    ]
                    -
                    result[
                        "balanced_accuracy_ai"
                    ],

                "Macro F1 AI":
                    result[
                        "macro_f1_ai"
                    ],

                "Macro F1 Oracle":
                    result[
                        "macro_f1_oracle"
                    ],

                "Delta Macro F1":
                    result[
                        "macro_f1_oracle"
                    ]
                    -
                    result[
                        "macro_f1_ai"
                    ],
            }

            per_seed_rows.append(
                row
            )

            print(
                f"{exp['name']} "
                f"seed={seed} | "
                f"F1 AI="
                f"{row['Macro F1 AI']:.4f} | "
                f"Oracle="
                f"{row['Macro F1 Oracle']:.4f} | "
                f"Delta="
                f"{row['Delta Macro F1']:+.4f}"
            )

    # =====================================================
    # Save per-seed
    # =====================================================

    df = pd.DataFrame(
        per_seed_rows
    )

    out_dir = Path(
        paths["results_dir"]
    ) / "contribution_2"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        out_dir /
        "c2_intervention_seed_results.csv",
        index=False
    )

    # =====================================================
    # Summary
    # =====================================================

    summary_rows = []

    for model_name in df[
        "Model"
    ].unique():

        sub = df[
            df["Model"] == model_name
        ]

        summary_rows.append({
            "Model":
                model_name,

            "Macro F1 AI":
                mean_sd(
                    sub[
                        "Macro F1 AI"
                    ]
                ),

            "Macro F1 Oracle":
                mean_sd(
                    sub[
                        "Macro F1 Oracle"
                    ]
                ),

            "Mean Delta Macro F1":
                f"{sub['Delta Macro F1'].mean():+.4f}",

            "Balanced Accuracy AI":
                mean_sd(
                    sub[
                        "Balanced Accuracy AI"
                    ]
                ),

            "Balanced Accuracy Oracle":
                mean_sd(
                    sub[
                        "Balanced Accuracy Oracle"
                    ]
                ),

            "Mean Delta Balanced Accuracy":
                f"{sub['Delta Balanced Accuracy'].mean():+.4f}",
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        out_dir /
        "c2_intervention_summary.csv",
        index=False
    )

    md_path = (
        out_dir /
        "c2_intervention_results.md"
    )

    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# C2 Oracle Concept "
            "Intervention\n\n"
        )

        f.write(
            "Ground-truth Derm7pt "
            "concept labels are substituted "
            "as soft oracle probabilities "
            f"({Config.ORACLE_NEG_PROB:.2f}/"
            f"{Config.ORACLE_POS_PROB:.2f}). "
            "This is an oracle diagnostic "
            "analysis, not a prospective "
            "clinician intervention study.\n\n"
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
            df.to_markdown(
                index=False,
                floatfmt=".4f"
            )
        )

    print(
        "\n=== C2 INTERVENTION SUMMARY ==="
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
