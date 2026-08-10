import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_sd(values, nan_safe=False):
    arr = np.asarray(values, dtype=float)
    if nan_safe:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return "N/A"
        return f"{np.mean(valid):.4f} ± {sample_sd(valid):.4f}"
    return f"{np.mean(arr):.4f} ± {sample_sd(arr):.4f}"


def plot_normalized_confusion_matrix(cm, classes, save_path, title):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_single_model(model, data_loader, device, num_classes):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.inference_mode():
        for batch in data_loader:
            clinic_img = batch["clinic_img"].to(device, non_blocking=True)
            derm_img = batch["derm_img"].to(device, non_blocking=True)
            meta_features = batch["metadata"].to(
                device, non_blocking=True
            )

            labels = batch["label_disease"].cpu().numpy()
            disease_logits, _ = model(
                clinic_img,
                derm_img,
                meta_features=meta_features,
            )

            probs = torch.softmax(disease_logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)
    labels = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    try:
        auroc = roc_auc_score(
            y_true,
            y_prob,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auroc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sensitivity, specificity = [], []
    for i in labels:
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity.append(
            tp / (tp + fn) if (tp + fn) > 0 else np.nan
        )
        specificity.append(
            tn / (tn + fp) if (tn + fp) > 0 else np.nan
        )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_specificity": float(np.nanmean(specificity)),
        "auroc": float(auroc),
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _require_all_checkpoints(paths):
    missing = []
    for exp in Config.PAPER_EXPERIMENTS:
        for seed in Config.SEEDS:
            path = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth",
            )
            if not os.path.exists(path):
                missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Thiếu paper checkpoints:\n- " + "\n- ".join(missing)
        )


def main():
    paths = Config.ensure_runtime_dirs()
    _require_all_checkpoints(paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    required = [
        paths["test_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file evaluation:\n- " + "\n- ".join(missing)
        )

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

    with open(paths["label_mapping"], "r", encoding="utf-8") as f:
        disease_to_idx = json.load(f)

    disease_names = [
        name
        for name, _ in sorted(
            disease_to_idx.items(), key=lambda item: item[1]
        )
    ]

    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    workers = 2 if paths["data_root"].startswith("/content/") else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    summary_rows = []
    per_seed_rows = []
    per_class_rows = []

    for exp in Config.PAPER_EXPERIMENTS:
        metric_lists = {
            "accuracy": [],
            "balanced_accuracy": [],
            "macro_f1": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_specificity": [],
            "auroc": [],
        }
        sens_by_class = [[] for _ in disease_names]
        spec_by_class = [[] for _ in disease_names]
        aggregated_cm = np.zeros(
            (Config.NUM_CLASSES, Config.NUM_CLASSES),
            dtype=float,
        )

        print(f"\nEvaluating {exp['name']} on independent Test set")

        for seed in Config.SEEDS:
            model_path = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth",
            )

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality=exp["modality"],
                bottleneck_type=exp["bottleneck"],
                use_metadata=exp["meta"],
                meta_input_dim=meta_input_dim,
            ).to(device)

            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state, strict=True)

            res = evaluate_single_model(
                model,
                test_loader,
                device,
                Config.NUM_CLASSES,
            )

            for key in metric_lists:
                metric_lists[key].append(res[key])

            aggregated_cm += res["cm"]

            for i in range(Config.NUM_CLASSES):
                sens_by_class[i].append(res["sensitivity"][i])
                spec_by_class[i].append(res["specificity"][i])

            per_seed_rows.append(
                {
                    "Model": exp["name"],
                    "Seed": seed,
                    "Accuracy": res["accuracy"],
                    "Balanced Accuracy": res["balanced_accuracy"],
                    "Macro F1": res["macro_f1"],
                    "Macro Precision": res["macro_precision"],
                    "Macro Recall": res["macro_recall"],
                    "Macro Specificity": res["macro_specificity"],
                    "One-vs-Rest AUROC": res["auroc"],
                }
            )

            print(
                f"  seed={seed} | "
                f"F1={res['macro_f1']:.4f} | "
                f"BAcc={res['balanced_accuracy']:.4f} | "
                f"AUROC={res['auroc']:.4f}"
            )

        summary_rows.append(
            {
                "Model": exp["name"],
                "Seeds": ",".join(map(str, Config.SEEDS)),
                "Accuracy": mean_sd(metric_lists["accuracy"]),
                "Balanced Accuracy": mean_sd(
                    metric_lists["balanced_accuracy"]
                ),
                "Macro F1": mean_sd(metric_lists["macro_f1"]),
                "Macro Precision": mean_sd(
                    metric_lists["macro_precision"]
                ),
                "Macro Recall": mean_sd(metric_lists["macro_recall"]),
                "Macro Specificity": mean_sd(
                    metric_lists["macro_specificity"]
                ),
                "One-vs-Rest AUROC": mean_sd(
                    metric_lists["auroc"], nan_safe=True
                ),
            }
        )

        for i, class_name in enumerate(disease_names):
            per_class_rows.append(
                {
                    "Model": exp["name"],
                    "Class": class_name,
                    "Sensitivity": mean_sd(
                        sens_by_class[i], nan_safe=True
                    ),
                    "Specificity": mean_sd(
                        spec_by_class[i], nan_safe=True
                    ),
                }
            )

        cm_path = os.path.join(
            paths["output_dir"],
            f"{exp['name']}_aggregated_normalized_cm.png",
        )
        plot_normalized_confusion_matrix(
            aggregated_cm,
            disease_names,
            cm_path,
            f"Aggregated Normalized Confusion Matrix - {exp['name']}",
        )

    summary_df = pd.DataFrame(summary_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    summary_df.to_csv(
        os.path.join(
            paths["output_dir"],
            "final_advanced_results_summary.csv",
        ),
        index=False,
    )
    per_seed_df.to_csv(
        os.path.join(paths["output_dir"], "per_seed_test_metrics.csv"),
        index=False,
    )
    per_class_df.to_csv(
        os.path.join(paths["output_dir"], "per_class_test_metrics.csv"),
        index=False,
    )

    final_md = os.path.join(paths["results_dir"], "final_results.md")
    with open(final_md, "w", encoding="utf-8") as f:
        f.write("# Final Test Results\n\n")
        f.write(
            "All model/hyperparameter selection is completed on Validation. "
            "The independent Test split is used only for final evaluation. "
            "Values are mean ± sample SD across the three locked seeds.\n\n"
        )

        try:
            selected_alpha = Config.load_selected_alpha()
            f.write(f"Selected concept-loss alpha: **{selected_alpha:.1f}**.\n\n")
        except Exception:
            pass

        f.write("## Overall metrics\n\n")
        f.write(summary_df.to_markdown(index=False))

        f.write("\n\n## Per-seed metrics\n\n")
        f.write(per_seed_df.to_markdown(index=False, floatfmt=".4f"))

        f.write("\n\n## Class-wise sensitivity/specificity\n\n")
        f.write(per_class_df.to_markdown(index=False))

    print("\nFINAL SUMMARY")
    print(summary_df.to_markdown(index=False))
    print(f"\nSaved: {final_md}")


if __name__ == "__main__":
    main()