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
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.4f} ± {sample_sd(arr):.4f}"


def extract_features(model, loader, device):
    true_disease = []
    true_concepts = []
    pred_concepts = []
    metadata = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            c_img = batch["clinic_img"].to(device, non_blocking=True)
            d_img = batch["derm_img"].to(device, non_blocking=True)
            meta = batch["metadata"].to(device, non_blocking=True)

            _, concept_logits = model(
                c_img,
                d_img,
                meta_features=meta,
            )
            if concept_logits is None:
                raise RuntimeError("B6 PureCBM phải có concept head.")

            concept_probs = torch.sigmoid(concept_logits)

            true_disease.extend(
                batch["label_disease"].cpu().numpy()
            )
            true_concepts.extend(
                batch["concept_labels"].cpu().numpy()
            )
            pred_concepts.extend(
                concept_probs.cpu().numpy()
            )
            metadata.extend(meta.cpu().numpy())

    return (
        np.asarray(true_disease),
        np.asarray(true_concepts),
        np.asarray(pred_concepts),
        np.asarray(metadata),
    )


def _soft_oracle_np(concepts):
    return np.where(
        concepts > 0.5,
        Config.ORACLE_POS_PROB,
        Config.ORACLE_NEG_PROB,
    ).astype(np.float32)


def main():
    """
    Sequential concept-quality gap analysis.

    For each B6 seed, two downstream logistic classifiers are trained
    using TRAIN only:

    1) Predicted-concept sequential classifier:
       train predicted concept probabilities + metadata -> disease.
       It is evaluated on Test predicted concepts.

    2) Oracle-concept upper bound:
       train soft ground-truth concepts + metadata -> disease.
       It is evaluated on Test soft ground-truth concepts.

    The difference estimates how much downstream performance is limited
    by concept prediction quality without feeding a classifier inputs from
    a distribution it was not trained on.
    """
    paths = Config.ensure_runtime_dirs()

    if os.path.isdir("/content/local_images"):
        paths["img_dir"] = "/content/local_images"

    print("Image source:", paths["img_dir"])

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_paths = [
        os.path.join(
            paths["output_dir"], f"B6_PureCBM_seed_{seed}.pth"
        )
        for seed in Config.SEEDS
    ]
    missing = [p for p in checkpoint_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu B6 checkpoint:\n- " + "\n- ".join(missing)
        )

    per_seed_rows = []
    pred_f1_all, oracle_f1_all = [], []
    pred_acc_all, oracle_acc_all = [], []

    class_labels = list(range(Config.NUM_CLASSES))

    for seed in Config.SEEDS:
        model_path = os.path.join(
            paths["output_dir"], f"B6_PureCBM_seed_{seed}.pth"
        )

        model = MultimodalDermModel(
            num_classes=Config.NUM_CLASSES,
            num_concepts=Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="pure",
            use_metadata=True,
            meta_input_dim=meta_input_dim,
        ).to(device)

        model.load_state_dict(
            torch.load(model_path, map_location=device),
            strict=True,
        )

        (
            y_train_d,
            y_train_true_c,
            x_train_pred_c,
            x_train_meta,
        ) = extract_features(model, train_loader, device)

        (
            y_test_d,
            y_test_true_c,
            x_test_pred_c,
            x_test_meta,
        ) = extract_features(model, test_loader, device)

        # A. Realistic sequential classifier: trained and tested on
        # predicted concept probability distributions.
        x_train_pred = x_train_pred_c
        x_test_pred = x_test_pred_c

        clf_pred = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf_pred.fit(x_train_pred, y_train_d)
        pred_from_predicted = clf_pred.predict(x_test_pred)

        # B. Oracle upper bound: train/test both use the same soft-oracle
        # concept representation, avoiding a train-test concept shift.
        x_train_oracle = _soft_oracle_np(
            y_train_true_c
        )
        x_test_oracle = _soft_oracle_np(
            y_test_true_c
        )

        clf_oracle = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        clf_oracle.fit(x_train_oracle, y_train_d)
        pred_from_oracle = clf_oracle.predict(x_test_oracle)

        pred_f1 = f1_score(
            y_test_d,
            pred_from_predicted,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        oracle_f1 = f1_score(
            y_test_d,
            pred_from_oracle,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        pred_acc = accuracy_score(y_test_d, pred_from_predicted)
        oracle_acc = accuracy_score(y_test_d, pred_from_oracle)

        pred_f1_all.append(pred_f1)
        oracle_f1_all.append(oracle_f1)
        pred_acc_all.append(pred_acc)
        oracle_acc_all.append(oracle_acc)

        per_seed_rows.append(
            {
                "Seed": seed,
                "Accuracy Predicted Concepts": pred_acc,
                "Accuracy Oracle Concepts": oracle_acc,
                "Macro F1 Predicted Concepts": pred_f1,
                "Macro F1 Oracle Concepts": oracle_f1,
                "Oracle Gap Macro F1": oracle_f1 - pred_f1,
            }
        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = pd.DataFrame(
        [
            {
                "Seed": "Mean ± sample SD",
                "Accuracy Predicted Concepts": mean_sd(pred_acc_all),
                "Accuracy Oracle Concepts": mean_sd(oracle_acc_all),
                "Macro F1 Predicted Concepts": mean_sd(pred_f1_all),
                "Macro F1 Oracle Concepts": mean_sd(oracle_f1_all),
                "Oracle Gap Macro F1": (
                    f"{np.mean(np.asarray(oracle_f1_all) - np.asarray(pred_f1_all)):+.4f}"
                ),
            }
        ]
    )

    out_dir = os.path.join(
        paths["results_dir"],
        "contribution_2"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    out_path = os.path.join(
        out_dir,
        "c2_sequential_cbm_results.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Sequential CBM Concept-Quality Gap Analysis\n\n")
        f.write(
            "Two downstream classifiers are trained using Train only: "
            "one on predicted concept probabilities and one on soft "
            "ground-truth concepts (oracle upper bound). Test is used only "
            "for final evaluation. This is an oracle analysis, not a "
            "prospective clinician study.\n\n"
        )
        f.write("## Per-seed results\n\n")
        f.write(per_seed_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

    print(per_seed_df.to_markdown(index=False, floatfmt=".4f"))
    print("\n" + summary_df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()