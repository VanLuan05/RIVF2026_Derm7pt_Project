import json
import os
from itertools import combinations

import pandas as pd

from src.config import Config
from src.data.dataset import get_real_path


def _load_mapping(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing label mapping: {path}")

    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    ids = sorted(int(v) for v in mapping.values())
    expected = list(range(Config.NUM_CLASSES))
    if ids != expected:
        raise RuntimeError(
            f"Label mapping IDs phải là {expected}, nhưng nhận {ids}."
        )

    if len(mapping) != Config.NUM_CLASSES:
        raise RuntimeError(
            f"Expected {Config.NUM_CLASSES} classes, got {len(mapping)}."
        )

    return mapping


def _audit_images(dfs, img_dir):
    missing_rows = []
    checked = {}

    for split_name, df in dfs.items():
        for column in ("clinic", "derm"):
            for rel_path in df[column].astype(str).unique():
                key = (column, rel_path)
                if key not in checked:
                    checked[key] = get_real_path(img_dir, rel_path)

                real_path = checked[key]
                if real_path is None or not os.path.isfile(real_path):
                    missing_rows.append(
                        {
                            "Split": split_name,
                            "Modality": column,
                            "Relative path": rel_path,
                        }
                    )

    return pd.DataFrame(missing_rows), len(checked)


def main():
    paths = Config.ensure_runtime_dirs()
    mapping = _load_mapping(paths["label_mapping"])

    split_paths = {
        "Train": paths["train_csv"],
        "Validation": paths["val_csv"],
        "Calibration": paths["calib_csv"],
        "Test": paths["test_csv"],
    }

    required_columns = {
        "case_num",
        "standard_diagnosis",
        "clinic",
        "derm",
        "sex",
        "location",
        "elevation",
    }

    dfs = {}
    distribution_rows = []
    group_rows = []
    fatal_errors = []

    class_order = [
        name
        for name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]

    for split_name, csv_path in split_paths.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split file: {csv_path}")

        df = pd.read_csv(csv_path)

        missing_cols = sorted(required_columns - set(df.columns))
        if missing_cols:
            raise RuntimeError(
                f"{split_name} thiếu cột bắt buộc: {missing_cols}"
            )

        dfs[split_name] = df

        counts = (
            df["standard_diagnosis"]
            .value_counts()
            .reindex(class_order, fill_value=0)
        )

        for cls in class_order:
            count = int(counts.loc[cls])
            distribution_rows.append(
                {
                    "Split": split_name,
                    "Class": cls,
                    "Count": count,
                    "Percent": 100.0 * count / max(len(df), 1),
                }
            )

        # Train/Validation/Test must contain all five classes for the locked
        # classification metrics. Calibration is currently reserved.
        if split_name in {"Train", "Validation", "Test"}:
            absent = [cls for cls in class_order if counts.loc[cls] == 0]
            if absent:
                fatal_errors.append(
                    f"{split_name} thiếu class: {absent}"
                )

        case_series = df["case_num"]
        unknown_mask = (
            case_series.isna()
            | case_series.astype(str)
            .str.strip()
            .str.lower()
            .isin({"", "unknown_case", "nan", "none"})
        )

        group_rows.append(
            {
                "Split": split_name,
                "Samples": int(len(df)),
                "Unique case_num": int(
                    case_series.astype(str).nunique()
                ),
                "Unknown/missing case_num rows": int(unknown_mask.sum()),
            }
        )

        if unknown_mask.any():
            fatal_errors.append(
                f"{split_name} có {int(unknown_mask.sum())} "
                "row thiếu/unknown case_num."
            )

    dist_df = pd.DataFrame(distribution_rows)
    group_df = pd.DataFrame(group_rows)

    overlap_rows = []
    for a, b in combinations(dfs.keys(), 2):
        cases_a = set(dfs[a]["case_num"].astype(str))
        cases_b = set(dfs[b]["case_num"].astype(str))
        overlap = sorted(cases_a.intersection(cases_b))
        overlap_rows.append(
            {
                "Split A": a,
                "Split B": b,
                "Overlapping case_num": len(overlap),
            }
        )
        if overlap:
            fatal_errors.append(
                f"{a} và {b} overlap {len(overlap)} case_num."
            )

    overlap_df = pd.DataFrame(overlap_rows)

    # Audit every referenced image before expensive GPU training.
    if not os.path.isdir(paths["img_dir"]):
        fatal_errors.append(
            f"Image directory không tồn tại: {paths['img_dir']}"
        )
        missing_img_df = pd.DataFrame()
        checked_image_paths = 0
    else:
        missing_img_df, checked_image_paths = _audit_images(
            dfs, paths["img_dir"]
        )
        if not missing_img_df.empty:
            fatal_errors.append(
                f"Có {len(missing_img_df)} image references không tìm thấy."
            )

    out_path = os.path.join(paths["results_dir"], "split_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Split and Data Audit\n\n")
        f.write(
            "The split is audited at `case_num` group level. "
            "Do not describe it as patient-level unless Derm7pt "
            "documentation independently confirms that `case_num` "
            "uniquely identifies patients.\n\n"
        )

        f.write("## Class distribution\n\n")
        f.write(dist_df.to_markdown(index=False, floatfmt=".2f"))

        f.write("\n\n## case_num overlap\n\n")
        f.write(overlap_df.to_markdown(index=False))

        f.write("\n\n## Group summary\n\n")
        f.write(group_df.to_markdown(index=False))

        f.write("\n\n## Image audit\n\n")
        f.write(f"Unique referenced image paths checked: {checked_image_paths}\n\n")
        if missing_img_df.empty:
            f.write("No missing referenced images were detected.\n")
        else:
            f.write(
                missing_img_df.head(100).to_markdown(index=False)
            )
            if len(missing_img_df) > 100:
                f.write(
                    f"\n\nOnly first 100/{len(missing_img_df)} "
                    "missing references are shown."
                )

        f.write("\n\n## Final status\n\n")
        if fatal_errors:
            for err in fatal_errors:
                f.write(f"- FAIL: {err}\n")
        else:
            f.write("- PASS: no blocking split/data integrity issue detected.\n")

    print("\nCLASS DISTRIBUTION")
    print(dist_df.to_markdown(index=False, floatfmt=".2f"))
    print("\nCASE OVERLAP")
    print(overlap_df.to_markdown(index=False))
    print("\nGROUP SUMMARY")
    print(group_df.to_markdown(index=False))
    print(
        f"\nImage references checked: {checked_image_paths}; "
        f"missing: {len(missing_img_df)}"
    )
    print(f"Saved: {out_path}")

    if fatal_errors:
        raise RuntimeError(
            "DATA AUDIT FAILED:\n- " + "\n- ".join(fatal_errors)
        )

    print("\n[PASS] Data/split audit passed. Có thể chạy alpha ablation.")


if __name__ == "__main__":
    main()