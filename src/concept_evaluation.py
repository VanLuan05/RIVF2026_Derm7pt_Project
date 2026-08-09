import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")

CONCEPT_NAMES = [
    "Pigment Network",
    "Streaks",
    "Pigmentation",
    "Regression Structures",
    "Dots & Globules",
    "Blue-whitish Veil",
    "Vascular Structures",
]


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def fmt(values):
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return "N/A"
    return f"{np.mean(valid):.4f} ± {sample_sd(valid):.4f}"


def collect_concepts(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            c_img = batch["clinic_img"].to(device)
            d_img = batch["derm_img"].to(device)
            meta = batch["metadata"].to(device)
            _, concept_logits = model(c_img, d_img, meta_features=meta)
            if concept_logits is None:
                raise RuntimeError("Model không có concept head.")
            y_true.append(batch["concept_labels"].cpu().numpy())
            y_prob.append(torch.sigmoid(concept_logits).cpu().numpy())
    return np.vstack(y_true), np.vstack(y_prob)


def main():
    paths = Config.runtime_paths()
    os.makedirs(paths["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(paths["meta_encoder"]):
        raise FileNotFoundError("Thiếu meta_encoder.joblib.")

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
    loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
    )

    experiments = [
        {"name": "B6_PureCBM", "bottleneck": "pure"},
        {"name": "Proposed_Hybrid", "bottleneck": "hybrid"},
    ]

    rows = []
    for exp in experiments:
        per_concept = {
            name: {"auroc": [], "f1": [], "precision": [], "recall": []}
            for name in CONCEPT_NAMES
        }
        prevalence = None

        for seed in Config.SEEDS:
            path = os.path.join(paths["output_dir"], f"{exp['name']}_seed_{seed}.pth")
            if not os.path.exists(path):
                print(f"[skip] Missing {path}")
                continue

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type=exp["bottleneck"],
                use_metadata=True,
                meta_input_dim=meta_input_dim,
            ).to(device)
            model.load_state_dict(torch.load(path, map_location=device), strict=True)

            y_true, y_prob = collect_concepts(model, loader, device)
            y_pred = (y_prob >= 0.5).astype(int)
            prevalence = y_true.mean(axis=0)

            for j, name in enumerate(CONCEPT_NAMES):
                if len(np.unique(y_true[:, j])) == 2:
                    auroc = roc_auc_score(y_true[:, j], y_prob[:, j])
                else:
                    auroc = np.nan
                per_concept[name]["auroc"].append(auroc)
                per_concept[name]["f1"].append(
                    f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
                )
                per_concept[name]["precision"].append(
                    precision_score(y_true[:, j], y_pred[:, j], zero_division=0)
                )
                per_concept[name]["recall"].append(
                    recall_score(y_true[:, j], y_pred[:, j], zero_division=0)
                )

        for j, name in enumerate(CONCEPT_NAMES):
            if not per_concept[name]["f1"]:
                continue
            rows.append(
                {
                    "Model": exp["name"],
                    "Concept": name,
                    "Test Prevalence": f"{prevalence[j]:.4f}",
                    "AUROC": fmt(per_concept[name]["auroc"]),
                    "F1": fmt(per_concept[name]["f1"]),
                    "Precision": fmt(per_concept[name]["precision"]),
                    "Recall": fmt(per_concept[name]["recall"]),
                }
            )

    df = pd.DataFrame(rows)
    out_path = os.path.join(paths["results_dir"], "concept_metrics.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Concept Prediction Performance on Test Set\n\n")
        f.write(
            "Threshold-dependent metrics use a fixed 0.5 threshold. AUROC is threshold-free. "
            "Values are mean ± sample SD across available seeds.\n\n"
        )
        f.write(df.to_markdown(index=False))

    print(df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()