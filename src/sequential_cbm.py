import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_sd(values):
    return f"{np.mean(values):.4f} ± {sample_sd(values):.4f}"


def extract_features(model, loader, device):
    true_disease, true_concepts, pred_concepts, metadata = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            c_img = batch["clinic_img"].to(device)
            d_img = batch["derm_img"].to(device)
            meta = batch["metadata"].to(device)
            _, concept_logits = model(c_img, d_img, meta_features=meta)
            concept_probs = torch.sigmoid(concept_logits)

            true_disease.extend(batch["label_disease"].cpu().numpy())
            true_concepts.extend(batch["concept_labels"].cpu().numpy())
            pred_concepts.extend(concept_probs.cpu().numpy())
            metadata.extend(meta.cpu().numpy())

    return (
        np.asarray(true_disease),
        np.asarray(true_concepts),
        np.asarray(pred_concepts),
        np.asarray(metadata),
    )


def main():
    paths = Config.runtime_paths()
    os.makedirs(paths["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())
    workers = 2 if paths["data_root"].startswith("/content/") else 0

    train_dataset = MultimodalDermDataset(
        paths["train_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )
    test_dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=workers)

    seed_rows = []
    f1_before_all, f1_after_all = [], []
    acc_before_all, acc_after_all = [], []

    for seed in Config.SEEDS:
        model_path = os.path.join(paths["output_dir"], f"B6_PureCBM_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"[skip] Missing {model_path}")
            continue

        model = MultimodalDermModel(
            num_classes=Config.NUM_CLASSES,
            num_concepts=Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="pure",
            use_metadata=True,
            meta_input_dim=meta_input_dim,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

        y_train_d, y_train_c, _, x_train_meta = extract_features(model, train_loader, device)
        y_test_d, y_test_c, x_test_pred_c, x_test_meta = extract_features(model, test_loader, device)

        # Sequential CBM: disease classifier is trained on ground-truth concepts + metadata.
        x_train_seq = np.hstack([y_train_c, x_train_meta])
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(x_train_seq, y_train_d)

        x_test_ai = np.hstack([x_test_pred_c, x_test_meta])
        x_test_oracle = np.hstack([y_test_c, x_test_meta])
        pred_before = clf.predict(x_test_ai)
        pred_after = clf.predict(x_test_oracle)

        f1_before = f1_score(y_test_d, pred_before, average="macro", zero_division=0)
        f1_after = f1_score(y_test_d, pred_after, average="macro", zero_division=0)
        acc_before = accuracy_score(y_test_d, pred_before)
        acc_after = accuracy_score(y_test_d, pred_after)

        f1_before_all.append(f1_before)
        f1_after_all.append(f1_after)
        acc_before_all.append(acc_before)
        acc_after_all.append(acc_after)

        seed_rows.append(
            {
                "Seed": seed,
                "Accuracy AI Concepts": acc_before,
                "Accuracy Oracle Concepts": acc_after,
                "Macro F1 AI Concepts": f1_before,
                "Macro F1 Oracle Concepts": f1_after,
                "Delta Macro F1": f1_after - f1_before,
            }
        )

    if not seed_rows:
        raise RuntimeError("Không có B6 checkpoint nào để chạy Sequential CBM.")

    summary_row = {
        "Seed": "Mean ± sample SD",
        "Accuracy AI Concepts": mean_sd(acc_before_all),
        "Accuracy Oracle Concepts": mean_sd(acc_after_all),
        "Macro F1 AI Concepts": mean_sd(f1_before_all),
        "Macro F1 Oracle Concepts": mean_sd(f1_after_all),
        "Delta Macro F1": f"{np.mean(np.asarray(f1_after_all) - np.asarray(f1_before_all)):+.4f}",
    }

    seed_df = pd.DataFrame(seed_rows)
    summary_df = pd.DataFrame([summary_row])
    out_path = os.path.join(paths["results_dir"], "sequential_cbm_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Sequential CBM Oracle-Concept Analysis\n\n")
        f.write(
            "This is an oracle analysis using dataset ground-truth concepts; it is not a prospective doctor study.\n\n"
        )
        f.write("## Per-seed results\n\n")
        f.write(seed_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

    print(seed_df.to_markdown(index=False, floatfmt=".4f"))
    print("\n" + summary_df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()