
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import Config


def safe_auroc(y_true, score):
    try:
        return float(
            roc_auc_score(
                y_true,
                score
            )
        )
    except ValueError:
        return np.nan


def main():

    paths = Config.ensure_runtime_dirs()

    input_path = (
        Path(paths["results_dir"])
        / "contribution_5"
        / "c5_ensemble_predictions.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing: {input_path}"
        )

    df = pd.read_csv(input_path)

    # 1 = prediction error
    df["error"] = 1 - df["correct"]

    print(
        "Samples:",
        len(df)
    )

    print(
        "Correct:",
        int(df["correct"].sum())
    )

    print(
        "Incorrect:",
        int(df["error"].sum())
    )

    # ==========================================
    # Correct vs incorrect uncertainty
    # ==========================================

    metrics = [
        "confidence",
        "normalized_entropy",
        "margin",
        "seed_disagreement_rate",
        "mean_probability_sd",
        "max_probability_sd",
        "mutual_information",
    ]

    rows = []

    for metric in metrics:

        correct_values = df.loc[
            df["correct"] == 1,
            metric
        ]

        incorrect_values = df.loc[
            df["correct"] == 0,
            metric
        ]

        rows.append({
            "Metric": metric,
            "Correct_Mean":
                correct_values.mean(),
            "Correct_SD":
                correct_values.std(ddof=1),
            "Incorrect_Mean":
                incorrect_values.mean(),
            "Incorrect_SD":
                incorrect_values.std(ddof=1),
        })

    comparison_df = pd.DataFrame(rows)

    # ==========================================
    # Error-detection AUROC
    # Higher score must mean more uncertain
    # ==========================================

    uncertainty_scores = {
        "1-confidence":
            1.0 - df["confidence"],
        "normalized_entropy":
            df["normalized_entropy"],
        "1-margin":
            1.0 - df["margin"],
        "seed_disagreement_rate":
            df["seed_disagreement_rate"],
        "mean_probability_sd":
            df["mean_probability_sd"],
        "max_probability_sd":
            df["max_probability_sd"],
        "mutual_information":
            df["mutual_information"],
    }

    auroc_rows = []

    for name, score in uncertainty_scores.items():

        auroc_rows.append({
            "Uncertainty_Score": name,
            "Error_Detection_AUROC":
                safe_auroc(
                    df["error"],
                    score
                )
        })

    auroc_df = (
        pd.DataFrame(auroc_rows)
        .sort_values(
            "Error_Detection_AUROC",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # ==========================================
    # Top uncertain examples
    # ==========================================

    top_uncertain = (
        df.sort_values(
            [
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
        .head(20)
        .copy()
    )

    top_columns = [
        "sample_index",
        "case_num",
        "true_class",
        "pred_class",
        "correct",
        "confidence",
        "normalized_entropy",
        "margin",
        "seed_disagreement_rate",
        "mutual_information",
        "clinic",
        "derm",
    ]

    top_uncertain = top_uncertain[
        top_columns
    ]

    # ==========================================
    # Save
    # ==========================================

    out_dir = (
        Path(paths["results_dir"])
        / "contribution_5"
    )

    comparison_df.to_csv(
        out_dir
        / "c5_correct_vs_incorrect.csv",
        index=False
    )

    auroc_df.to_csv(
        out_dir
        / "c5_error_detection_auroc.csv",
        index=False
    )

    top_uncertain.to_csv(
        out_dir
        / "c5_top20_uncertain.csv",
        index=False
    )

    print(
        "\n=== CORRECT VS INCORRECT ==="
    )

    print(
        comparison_df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n=== ERROR DETECTION AUROC ==="
    )

    print(
        auroc_df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n=== TOP 20 UNCERTAIN ==="
    )

    print(
        top_uncertain[
            [
                "sample_index",
                "case_num",
                "true_class",
                "pred_class",
                "correct",
                "confidence",
                "normalized_entropy",
                "seed_disagreement_rate",
            ]
        ].to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n✅ C5 uncertainty analysis complete."
    )


if __name__ == "__main__":
    main()
