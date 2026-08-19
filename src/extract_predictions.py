
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

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


def entropy_from_probs(probs):
    probs = np.clip(
        probs,
        1e-12,
        1.0
    )

    return -np.sum(
        probs * np.log(probs),
        axis=1
    )


def normalized_entropy(probs):
    ent = entropy_from_probs(probs)

    return ent / np.log(
        probs.shape[1]
    )


def probability_margin(probs):
    sorted_probs = np.sort(
        probs,
        axis=1
    )

    return (
        sorted_probs[:, -1]
        - sorted_probs[:, -2]
    )


def extract_probabilities(
    model,
    loader,
    device
):
    all_labels = []
    all_probs = []

    model.eval()

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

            logits, _ = model(
                clinic,
                derm,
                meta_features=metadata
            )

            probs = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

            all_probs.extend(probs)

            all_labels.extend(
                batch[
                    "label_disease"
                ].cpu().numpy()
            )

    return (
        np.asarray(
            all_labels
        ),
        np.asarray(
            all_probs
        )
    )


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

    # ==========================================
    # Disease mapping
    # ==========================================
    with open(
        paths["label_mapping"],
        "r",
        encoding="utf-8"
    ) as f:
        disease_to_idx = json.load(f)

    idx_to_disease = {
        int(v): k
        for k, v
        in disease_to_idx.items()
    }

    # ==========================================
    # Metadata encoder
    # ==========================================
    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_input_dim = len(
        encoder.get_feature_names_out()
    )

    # ==========================================
    # Test dataset
    # ==========================================
    test_df = pd.read_csv(
        paths["test_csv"]
    ).reset_index(drop=True)

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=
            paths["meta_encoder"],
        transform=test_transforms,
    )

    workers = (
        2
        if paths["data_root"].startswith(
            "/content/"
        )
        else 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        "Test samples:",
        len(test_dataset)
    )

    # ==========================================
    # Proposed Hybrid × 3 seeds
    # ==========================================
    all_seed_probs = []
    seed_rows = []

    expected_labels = None

    for seed in Config.SEEDS:

        checkpoint = os.path.join(
            paths["output_dir"],
            f"Proposed_Hybrid_seed_{seed}.pth"
        )

        if not os.path.exists(
            checkpoint
        ):
            raise FileNotFoundError(
                f"Missing checkpoint: {checkpoint}"
            )

        model = MultimodalDermModel(
            num_classes=
                Config.NUM_CLASSES,
            num_concepts=
                Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="hybrid",
            use_metadata=True,
            meta_input_dim=
                meta_input_dim,
        ).to(device)

        state = torch.load(
            checkpoint,
            map_location=device
        )

        model.load_state_dict(
            state,
            strict=True
        )

        labels, probs = (
            extract_probabilities(
                model,
                test_loader,
                device
            )
        )

        if expected_labels is None:
            expected_labels = labels
        else:
            if not np.array_equal(
                expected_labels,
                labels
            ):
                raise RuntimeError(
                    "Test label order changed "
                    "between seeds."
                )

        if len(probs) != len(test_df):
            raise RuntimeError(
                "Prediction count does not "
                "match Test CSV."
            )

        all_seed_probs.append(
            probs
        )

        preds = np.argmax(
            probs,
            axis=1
        )

        confidence = np.max(
            probs,
            axis=1
        )

        entropy = entropy_from_probs(
            probs
        )

        norm_entropy = normalized_entropy(
            probs
        )

        margin = probability_margin(
            probs
        )

        for i in range(
            len(test_df)
        ):

            row = {
                "sample_index": i,
                "case_num":
                    test_df.iloc[i][
                        "case_num"
                    ],
                "case_id":
                    test_df.iloc[i].get(
                        "case_id",
                        np.nan
                    ),
                "clinic":
                    test_df.iloc[i][
                        "clinic"
                    ],
                "derm":
                    test_df.iloc[i][
                        "derm"
                    ],
                "seed": seed,
                "true_class_idx":
                    int(labels[i]),
                "true_class":
                    idx_to_disease[
                        int(labels[i])
                    ],
                "pred_class_idx":
                    int(preds[i]),
                "pred_class":
                    idx_to_disease[
                        int(preds[i])
                    ],
                "correct":
                    int(
                        preds[i]
                        == labels[i]
                    ),
                "confidence":
                    float(
                        confidence[i]
                    ),
                "entropy":
                    float(
                        entropy[i]
                    ),
                "normalized_entropy":
                    float(
                        norm_entropy[i]
                    ),
                "margin":
                    float(
                        margin[i]
                    ),
            }

            for c in range(
                Config.NUM_CLASSES
            ):
                row[
                    f"p_class_{c}"
                ] = float(
                    probs[i, c]
                )

            seed_rows.append(
                row
            )

        print(
            f"✅ seed={seed} extracted"
        )

    # shape:
    # seeds × samples × classes
    prob_stack = np.stack(
        all_seed_probs,
        axis=0
    )

    # ==========================================
    # Ensemble probability
    # Secondary uncertainty analysis
    # ==========================================
    ensemble_probs = np.mean(
        prob_stack,
        axis=0
    )

    ensemble_preds = np.argmax(
        ensemble_probs,
        axis=1
    )

    ensemble_conf = np.max(
        ensemble_probs,
        axis=1
    )

    ensemble_entropy = (
        entropy_from_probs(
            ensemble_probs
        )
    )

    ensemble_norm_entropy = (
        normalized_entropy(
            ensemble_probs
        )
    )

    ensemble_margin = (
        probability_margin(
            ensemble_probs
        )
    )

    # Seed-level predicted classes
    seed_predictions = np.argmax(
        prob_stack,
        axis=2
    )

    # Variation ratio:
    # 0 = all seeds agree
    # 2/3 = all predictions differ maximally
    disagreement = []

    for i in range(
        len(test_df)
    ):

        values, counts = np.unique(
            seed_predictions[:, i],
            return_counts=True
        )

        max_count = np.max(
            counts
        )

        variation_ratio = (
            1.0
            - max_count
            / len(Config.SEEDS)
        )

        disagreement.append(
            variation_ratio
        )

    disagreement = np.asarray(
        disagreement
    )

    # Probability SD across seeds
    prob_sd = np.std(
        prob_stack,
        axis=0,
        ddof=1
    )

    mean_prob_sd = np.mean(
        prob_sd,
        axis=1
    )

    max_prob_sd = np.max(
        prob_sd,
        axis=1
    )

    # Approximate ensemble mutual information:
    # predictive entropy
    # minus expected seed entropy
    seed_entropies = np.stack(
        [
            entropy_from_probs(
                prob_stack[s]
            )
            for s in range(
                len(Config.SEEDS)
            )
        ],
        axis=0
    )

    mean_seed_entropy = np.mean(
        seed_entropies,
        axis=0
    )

    mutual_information = (
        ensemble_entropy
        - mean_seed_entropy
    )

    mutual_information = np.maximum(
        mutual_information,
        0.0
    )

    # ==========================================
    # Ensemble per-sample rows
    # ==========================================
    ensemble_rows = []

    for i in range(
        len(test_df)
    ):

        row = {
            "sample_index": i,
            "case_num":
                test_df.iloc[i][
                    "case_num"
                ],
            "case_id":
                test_df.iloc[i].get(
                    "case_id",
                    np.nan
                ),
            "clinic":
                test_df.iloc[i][
                    "clinic"
                ],
            "derm":
                test_df.iloc[i][
                    "derm"
                ],
            "true_class_idx":
                int(
                    expected_labels[i]
                ),
            "true_class":
                idx_to_disease[
                    int(
                        expected_labels[i]
                    )
                ],
            "pred_class_idx":
                int(
                    ensemble_preds[i]
                ),
            "pred_class":
                idx_to_disease[
                    int(
                        ensemble_preds[i]
                    )
                ],
            "correct":
                int(
                    ensemble_preds[i]
                    == expected_labels[i]
                ),
            "confidence":
                float(
                    ensemble_conf[i]
                ),
            "entropy":
                float(
                    ensemble_entropy[i]
                ),
            "normalized_entropy":
                float(
                    ensemble_norm_entropy[i]
                ),
            "margin":
                float(
                    ensemble_margin[i]
                ),
            "seed_disagreement_rate":
                float(
                    disagreement[i]
                ),
            "mean_probability_sd":
                float(
                    mean_prob_sd[i]
                ),
            "max_probability_sd":
                float(
                    max_prob_sd[i]
                ),
            "mutual_information":
                float(
                    mutual_information[i]
                ),
        }

        for c in range(
            Config.NUM_CLASSES
        ):
            row[
                f"p_class_{c}"
            ] = float(
                ensemble_probs[i, c]
            )

        ensemble_rows.append(
            row
        )

    seed_df = pd.DataFrame(
        seed_rows
    )

    ensemble_df = pd.DataFrame(
        ensemble_rows
    )

    # ==========================================
    # Active-learning style uncertainty ranking
    # Retrospective only — Test is NOT retrained
    # ==========================================
    ranking_df = (
        ensemble_df
        .sort_values(
            by=[
                "normalized_entropy",
                "seed_disagreement_rate",
                "margin"
            ],
            ascending=[
                False,
                False,
                True
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranking_df.insert(
        0,
        "uncertainty_rank",
        np.arange(
            1,
            len(ranking_df) + 1
        )
    )

    # ==========================================
    # Summary
    # ==========================================
    summary_rows = []

    for seed in Config.SEEDS:

        sub = seed_df[
            seed_df[
                "seed"
            ] == seed
        ]

        summary_rows.append({
            "Model":
                "Proposed_Hybrid",
            "Source":
                f"seed_{seed}",
            "Accuracy":
                sub[
                    "correct"
                ].mean(),
            "Mean Confidence":
                sub[
                    "confidence"
                ].mean(),
            "Mean Normalized Entropy":
                sub[
                    "normalized_entropy"
                ].mean(),
            "Mean Margin":
                sub[
                    "margin"
                ].mean(),
        })

    summary_rows.append({
        "Model":
            "Proposed_Hybrid",
        "Source":
            "3-seed probability ensemble",
        "Accuracy":
            ensemble_df[
                "correct"
            ].mean(),
        "Mean Confidence":
            ensemble_df[
                "confidence"
            ].mean(),
        "Mean Normalized Entropy":
            ensemble_df[
                "normalized_entropy"
            ].mean(),
        "Mean Margin":
            ensemble_df[
                "margin"
            ].mean(),
    })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ==========================================
    # Save
    # ==========================================
    out_dir = Path(
        paths["results_dir"]
    ) / "contribution_5"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # README-compatible output
    ensemble_df.to_csv(
        Path(
            paths["output_dir"]
        ) / "test_predictions.csv",
        index=False
    )

    seed_df.to_csv(
        out_dir /
        "c5_seed_predictions.csv",
        index=False
    )

    ensemble_df.to_csv(
        out_dir /
        "c5_ensemble_predictions.csv",
        index=False
    )

    ranking_df.to_csv(
        out_dir /
        "c5_uncertainty_ranking.csv",
        index=False
    )

    summary_df.to_csv(
        out_dir /
        "c5_uncertainty_summary.csv",
        index=False
    )

    print(
        "\n=== C5 UNCERTAINTY SUMMARY ==="
    )

    print(
        summary_df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n✅ Saved:"
    )

    print(
        Path(
            paths["output_dir"]
        ) /
        "test_predictions.csv"
    )

    print(
        out_dir
    )


if __name__ == "__main__":
    main()
