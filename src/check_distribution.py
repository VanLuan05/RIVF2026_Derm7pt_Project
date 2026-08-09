import os

import pandas as pd

from src.config import Config


def main():
    paths = Config.runtime_paths()
    os.makedirs(paths["results_dir"], exist_ok=True)

    split_paths = {
        "Train": paths["train_csv"],
        "Validation": paths["val_csv"],
        "Calibration": paths["calib_csv"],
        "Test": paths["test_csv"],
    }

    dfs = {}
    distribution_rows = []

    for split_name, csv_path in split_paths.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")
        df = pd.read_csv(csv_path)
        dfs[split_name] = df

        label_col = "standard_diagnosis" if "standard_diagnosis" in df.columns else "diagnosis"
        counts = df[label_col].value_counts().sort_index()
        for cls, count in counts.items():
            distribution_rows.append({
                "Split": split_name,
                "Class": cls,
                "Count": int(count),
                "Percent": 100.0 * count / len(df),
            })

    dist_df = pd.DataFrame(distribution_rows)

    # Audit overlap theo case_num. Đây là case-level group audit, không tự suy ra patient-level.
    overlap_rows = []
    split_names = list(dfs.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            cases_a = set(dfs[a]["case_num"].astype(str))
            cases_b = set(dfs[b]["case_num"].astype(str))
            overlap = cases_a.intersection(cases_b)
            overlap_rows.append({
                "Split A": a,
                "Split B": b,
                "Overlapping case_num": len(overlap),
            })

    overlap_df = pd.DataFrame(overlap_rows)

    unknown_rows = []
    for split_name, df in dfs.items():
        case_values = df["case_num"].astype(str).str.lower()
        unknown_count = case_values.isin({"unknown_case", "nan", "none"}).sum()
        unknown_rows.append({
            "Split": split_name,
            "Samples": len(df),
            "Unique case_num": df["case_num"].astype(str).nunique(),
            "Unknown/missing case_num rows": int(unknown_count),
        })
    unknown_df = pd.DataFrame(unknown_rows)

    out_path = os.path.join(paths["results_dir"], "split_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Split Audit\n\n")
        f.write(
            "The split is audited at `case_num` group level. Do not call it patient-level unless "
            "the dataset documentation independently confirms that `case_num` uniquely identifies patients.\n\n"
        )
        f.write("## Class distribution\n\n")
        f.write(dist_df.to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n## case_num overlap\n\n")
        f.write(overlap_df.to_markdown(index=False))
        f.write("\n\n## Group summary\n\n")
        f.write(unknown_df.to_markdown(index=False))

    print(dist_df.to_markdown(index=False, floatfmt=".2f"))
    print("\n" + overlap_df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()