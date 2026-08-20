import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    csv_path = Path(
        "results/contribution_5/c5_selective_prediction.csv"
    )

    output_path = Path(
        "results/contribution_5/c5_selective_prediction_plot.png"
    )

    df = pd.read_csv(csv_path)

    plot_df = df[
        df["Uncertainty_Score"] == "1-margin"
    ].copy()

    plot_df = plot_df.sort_values(
        "Coverage",
        ascending=False
    )

    print(
        plot_df[
            ["Coverage", "Accuracy", "Macro_F1"]
        ]
    )

    coverage_percent = plot_df["Coverage"] * 100

    plt.figure(figsize=(8, 5))

    plt.plot(
        coverage_percent,
        plot_df["Accuracy"],
        marker="o",
        label="Accuracy"
    )

    plt.plot(
        coverage_percent,
        plot_df["Macro_F1"],
        marker="s",
        label="Macro-F1"
    )

    plt.title(
        "Selective prediction using 1 - margin as uncertainty score"
    )

    plt.xlabel("Coverage (%)")
    plt.ylabel("Score")

    plt.ylim(0.35, 0.90)

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()