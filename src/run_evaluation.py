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
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def mean_sd(values, nan_safe=False):
    arr = np.asarray(values, dtype=float)
    if nan_safe:
        mean = float(np.nanmean(arr))
        valid = arr[~np.isnan(arr)]
        sd = sample_sd(valid)
    else:
        mean = float(np.mean(arr))
        sd = sample_sd(arr)
    return f"{mean:.4f} ± {sd:.4f}"


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

    with torch.no_grad():
        for batch in data_loader:
            clinic_img = batch["clinic_img"].to(device)
            derm_img = batch["derm_img"].to(device)
            meta_features = batch["metadata"].to(device)
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

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    sensitivity, specificity = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity.append(tp / (tp + fn) if (tp + fn) else 0.0)
        specificity.append(tn / (tn + fp) if (tn + fp) else 0.0)

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "auroc": auroc,
        "macro_specificity": float(np.mean(specificity)),
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main():
    paths = Config.runtime_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["output_dir"], exist_ok=True)

    if not os.path.exists(paths["meta_encoder"]):
        raise FileNotFoundError("Thiếu meta_encoder.joblib. Hãy chạy prepare_data.py trước.")

    with open(paths["label_mapping"], "r", encoding="utf-8") as f:
        disease_to_idx = json.load(f)
    disease_names = [
        name for name, _ in sorted(disease_to_idx.items(), key=lambda item: item[1])
    ]

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

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
    )

    experiments = [
        {"name": "B1_Clinical_Only", "modality": "clinic_only", "bottleneck": "none", "meta": False},
        {"name": "B2_Derm_Only", "modality": "derm_only", "bottleneck": "none", "meta": False},
        {"name": "B3_Meta_Only", "modality": "meta_only", "bottleneck": "none", "meta": True},
        {"name": "B4_Dual_NoMeta", "modality": "dual", "bottleneck": "none", "meta": False},
        {"name": "B5_Dual_Metadata", "modality": "dual", "bottleneck": "none", "meta": True},
        {"name": "B6_PureCBM", "modality": "dual", "bottleneck": "pure", "meta": True},
        {"name": "Proposed_Hybrid", "modality": "dual", "bottleneck": "hybrid", "meta": True},
    ]

    summary_rows = []
    per_class_rows = []

    for exp in experiments:
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
        aggregated_cm = np.zeros((Config.NUM_CLASSES, Config.NUM_CLASSES), dtype=float)
        used_seeds = []

        print(f"\nEvaluating {exp['name']} on independent Test set")
        for seed in Config.SEEDS:
            model_path = os.path.join(paths["output_dir"], f"{exp['name']}_seed_{seed}.pth")
            if not os.path.exists(model_path):
                print(f"  [skip] Missing {model_path}")
                continue

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

            res = evaluate_single_model(model, test_loader, device, Config.NUM_CLASSES)
            used_seeds.append(seed)
            for key in metric_lists:
                metric_lists[key].append(res[key])
            aggregated_cm += res["cm"]
            for i in range(Config.NUM_CLASSES):
                sens_by_class[i].append(res["sensitivity"][i])
                spec_by_class[i].append(res["specificity"][i])

            print(
                f"  seed={seed} | F1={res['macro_f1']:.4f} | "
                f"BAcc={res['balanced_accuracy']:.4f} | AUROC={res['auroc']:.4f}"
            )

        if not used_seeds:
            continue

        summary_rows.append(
            {
                "Model": exp["name"],
                "Seeds": ",".join(map(str, used_seeds)),
                "Accuracy": mean_sd(metric_lists["accuracy"]),
                "Balanced Accuracy": mean_sd(metric_lists["balanced_accuracy"]),
                "Macro F1": mean_sd(metric_lists["macro_f1"]),
                "Macro Precision": mean_sd(metric_lists["macro_precision"]),
                "Macro Recall": mean_sd(metric_lists["macro_recall"]),
                "Macro Specificity": mean_sd(metric_lists["macro_specificity"]),
                "One-vs-Rest AUROC": mean_sd(metric_lists["auroc"], nan_safe=True),
            }
        )

        for i, class_name in enumerate(disease_names):
            per_class_rows.append(
                {
                    "Model": exp["name"],
                    "Class": class_name,
                    "Sensitivity": mean_sd(sens_by_class[i]),
                    "Specificity": mean_sd(spec_by_class[i]),
                }
            )

        cm_path = os.path.join(
            paths["output_dir"], f"{exp['name']}_aggregated_normalized_cm.png"
        )
        plot_normalized_confusion_matrix(
            aggregated_cm,
            disease_names,
            cm_path,
            f"Aggregated Normalized Confusion Matrix - {exp['name']}",
        )

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    summary_csv = os.path.join(paths["output_dir"], "final_advanced_results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    final_md = os.path.join(paths["results_dir"], "final_results.md")
    with open(final_md, "w", encoding="utf-8") as f:
        f.write("# Final Test Results\n\n")
        f.write(
            "All models are evaluated on the independent Test split after checkpoint selection on Validation only. "
            "Values are mean ± sample SD across independent random seeds.\n\n"
        )
        f.write(summary_df.to_markdown(index=False))

    per_class_md = os.path.join(paths["results_dir"], "per_class_results.md")
    with open(per_class_md, "w", encoding="utf-8") as f:
        f.write("# Per-class Sensitivity and Specificity\n\n")
        f.write(per_class_df.to_markdown(index=False))

    print("\n" + summary_df.to_markdown(index=False))
    print(f"\nSaved: {final_md}")
    print(f"Saved: {per_class_md}")


if __name__ == "__main__":
    main()