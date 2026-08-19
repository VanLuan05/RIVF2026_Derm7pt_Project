
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.config import Config


COVERAGES = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
]


def evaluate_score(
    df,
    score_name,
    uncertainty
):
    rows = []

    order = np.argsort(
        np.asarray(uncertainty)
    )

    n_total = len(df)

    for coverage in COVERAGES:

        n_keep = max(
            1,
            int(
                np.ceil(
                    coverage * n_total
                )
            )
        )

        keep_idx = order[:n_keep]

        sub = df.iloc[
            keep_idx
        ]

        y_true = sub[
            "true_class_idx"
        ].to_numpy()

        y_pred = sub[
            "pred_class_idx"
        ].to_numpy()

        acc = accuracy_score(
            y_true,
            y_pred
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=list(
                range(
                    Config.NUM_CLASSES
                )
            ),
            zero_division=0,
        )

        rows.append({
            "Uncertainty_Score":
                score_name,
            "Coverage":
                coverage,
            "Retained_Samples":
                n_keep,
            "Rejected_Samples":
                n_total - n_keep,
            "Accuracy":
                acc,
            "Macro_F1":
                macro_f1,
            "Error_Rate":
                1.0 - acc,
        })

    return rows


def main():

    paths = Config.ensure_runtime_dirs()

    path = (
        Path(
            paths["results_dir"]
        )
        / "contribution_5"
        / "c5_ensemble_predictions.csv"
    )

    df = pd.read_csv(path)

    scores = {
        "1-confidence":
            1.0 - df["confidence"],

        "1-margin":
            1.0 - df["margin"],

        "normalized_entropy":
            df["normalized_entropy"],
    }

    rows = []

    for name, uncertainty in scores.items():

        rows.extend(
            evaluate_score(
                df,
                name,
                uncertainty
            )
        )

    result = pd.DataFrame(
        rows
    )

    out_dir = (
        Path(
            paths["results_dir"]
        )
        / "contribution_5"
    )

    result.to_csv(
        out_dir
        / "c5_selective_prediction.csv",
        index=False
    )

    print(
        "\n=== SELECTIVE PREDICTION ==="
    )

    print(
        result.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    print(
        "\n✅ Selective prediction analysis complete."
    )


if __name__ == "__main__":
    main()
