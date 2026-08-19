
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
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

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


def sample_sd(values):
    arr = np.asarray(values, dtype=float)

    if len(arr) <= 1:
        return 0.0

    return float(
        np.std(arr, ddof=1)
    )


def mean_sd(values):
    arr = np.asarray(values, dtype=float)

    return (
        f"{np.mean(arr):.4f} ± "
        f"{sample_sd(arr):.4f}"
    )


def extract_concepts(
    model,
    loader,
    device,
):

    diseases = []
    gt_concepts = []
    predicted_probs = []

    model.eval()

    with torch.inference_mode():

        for batch in loader:

            clinic = batch[
                "clinic_img"
            ].to(device)

            derm = batch[
                "derm_img"
            ].to(device)

            metadata = batch[
                "metadata"
            ].to(device)

            _, concept_logits = model(
                clinic,
                derm,
                meta_features=metadata,
            )

            probabilities = torch.sigmoid(
                concept_logits
            )

            diseases.extend(
                batch[
                    "label_disease"
                ].cpu().numpy()
            )

            gt_concepts.extend(
                batch[
                    "concept_labels"
                ].cpu().numpy()
            )

            predicted_probs.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.asarray(diseases),
        np.asarray(gt_concepts),
        np.asarray(predicted_probs),
    )


def metrics(
    y_true,
    y_pred,
):

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Balanced Accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "Macro F1":
            f1_score(
                y_true,
                y_pred,
                labels=list(
                    range(
                        Config.NUM_CLASSES
                    )
                ),
                average="macro",
                zero_division=0,
            ),
    }


def main():

    paths = Config.ensure_runtime_dirs()

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

    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_dim = len(
        encoder.get_feature_names_out()
    )

    train_ds = MultimodalDermDataset(
        paths["train_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=
            paths["meta_encoder"],
        transform=test_transforms,
    )

    test_ds = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=
            paths["meta_encoder"],
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    rows = []

    for seed in Config.SEEDS:

        checkpoint = os.path.join(
            paths["output_dir"],
            f"B6_PureCBM_seed_{seed}.pth"
        )

        model = MultimodalDermModel(
            num_classes=
                Config.NUM_CLASSES,
            num_concepts=
                Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="pure",
            use_metadata=True,
            meta_input_dim=meta_dim,
        ).to(device)

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device
            ),
            strict=True,
        )

        (
            y_train,
            gt_train,
            prob_train,
        ) = extract_concepts(
            model,
            train_loader,
            device,
        )

        (
            y_test,
            gt_test,
            prob_test,
        ) = extract_concepts(
            model,
            test_loader,
            device,
        )

        # Fixed 0.5 threshold.
        # No Test tuning.
        hard_train = (
            prob_train >= 0.5
        ).astype(np.float32)

        hard_test = (
            prob_test >= 0.5
        ).astype(np.float32)

        gt_test = gt_test.astype(
            np.float32
        )

        # -------------------------------------------------
        # Train disease classifier ONLY on binary concepts
        # -------------------------------------------------

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )

        clf.fit(
            hard_train,
            y_train
        )

        # Same classifier
        pred_ai = clf.predict(
            hard_test
        )

        pred_intervention = clf.predict(
            gt_test
        )

        ai = metrics(
            y_test,
            pred_ai
        )

        intervention = metrics(
            y_test,
            pred_intervention
        )

        row = {
            "Seed": seed,

            "Macro F1 AI":
                ai["Macro F1"],

            "Macro F1 Intervention":
                intervention["Macro F1"],

            "Delta Macro F1":
                intervention["Macro F1"]
                - ai["Macro F1"],

            "Balanced Accuracy AI":
                ai["Balanced Accuracy"],

            "Balanced Accuracy Intervention":
                intervention[
                    "Balanced Accuracy"
                ],

            "Delta Balanced Accuracy":
                intervention[
                    "Balanced Accuracy"
                ]
                - ai[
                    "Balanced Accuracy"
                ],

            "Accuracy AI":
                ai["Accuracy"],

            "Accuracy Intervention":
                intervention[
                    "Accuracy"
                ],
        }

        rows.append(row)

        print(
            f"seed={seed} | "
            f"AI F1="
            f"{row['Macro F1 AI']:.4f} | "
            f"GT Intervention="
            f"{row['Macro F1 Intervention']:.4f} | "
            f"Delta="
            f"{row['Delta Macro F1']:+.4f}"
        )

    df = pd.DataFrame(rows)

    summary = pd.DataFrame([
        {
            "Model":
                "Hard-bottleneck Sequential CBM",

            "Macro F1 AI":
                mean_sd(
                    df["Macro F1 AI"]
                ),

            "Macro F1 Intervention":
                mean_sd(
                    df[
                        "Macro F1 Intervention"
                    ]
                ),

            "Mean Delta Macro F1":
                f"{df['Delta Macro F1'].mean():+.4f}",

            "Balanced Accuracy AI":
                mean_sd(
                    df[
                        "Balanced Accuracy AI"
                    ]
                ),

            "Balanced Accuracy Intervention":
                mean_sd(
                    df[
                        "Balanced Accuracy Intervention"
                    ]
                ),

            "Mean Delta Balanced Accuracy":
                f"{df['Delta Balanced Accuracy'].mean():+.4f}",
        }
    ])

    out_dir = os.path.join(
        paths["results_dir"],
        "contribution_2"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    df.to_csv(
        os.path.join(
            out_dir,
            "c2_hard_intervention_seed.csv"
        ),
        index=False,
    )

    summary.to_csv(
        os.path.join(
            out_dir,
            "c2_hard_intervention_summary.csv"
        ),
        index=False,
    )

    md = os.path.join(
        out_dir,
        "c2_hard_intervention_results.md"
    )

    with open(
        md,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# C2 Hard-Bottleneck "
            "Concept Intervention\n\n"
        )

        f.write(
            "A downstream disease classifier "
            "is trained using Train hard "
            "predicted concepts only. "
            "Test predictions use the same "
            "classifier. Full intervention "
            "replaces all seven predicted "
            "binary concepts with ground-truth "
            "binary concepts. Threshold 0.5 "
            "is fixed and is not tuned on Test."
            "\n\n"
        )

        f.write(
            "## Summary\n\n"
        )

        f.write(
            summary.to_markdown(
                index=False
            )
        )

        f.write(
            "\n\n## Per-seed\n\n"
        )

        f.write(
            df.to_markdown(
                index=False,
                floatfmt=".4f"
            )
        )

    print(
        "\n=== C2 HARD INTERVENTION SUMMARY ==="
    )

    print(
        summary.to_markdown(
            index=False
        )
    )

    print(
        "\nSaved:",
        md
    )


if __name__ == "__main__":
    main()
