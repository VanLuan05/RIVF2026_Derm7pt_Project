
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel


def mean_sd(x):
    x = np.asarray(x, dtype=float)
    sd = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return f"{np.mean(x):.4f} ± {sd:.4f}"


def extract(model, loader, device):

    disease = []
    gt_concepts = []
    predicted_probs = []

    model.eval()

    with torch.inference_mode():

        for batch in loader:

            clinic = batch["clinic_img"].to(device)
            derm = batch["derm_img"].to(device)
            meta = batch["metadata"].to(device)

            _, concept_logits = model(
                clinic,
                derm,
                meta_features=meta,
            )

            probs = torch.sigmoid(concept_logits)

            disease.extend(
                batch["label_disease"].cpu().numpy()
            )

            gt_concepts.extend(
                batch["concept_labels"].cpu().numpy()
            )

            predicted_probs.extend(
                probs.cpu().numpy()
            )

    return (
        np.asarray(disease),
        np.asarray(gt_concepts),
        np.asarray(predicted_probs),
    )


def evaluate_lr(
    x_train,
    y_train,
    x_test,
    y_test,
    seed,
):

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )

    clf.fit(
        x_train,
        y_train
    )

    pred = clf.predict(
        x_test
    )

    return (
        accuracy_score(
            y_test,
            pred
        ),
        f1_score(
            y_test,
            pred,
            labels=list(
                range(Config.NUM_CLASSES)
            ),
            average="macro",
            zero_division=0,
        ),
    )


def main():

    paths = Config.ensure_runtime_dirs()

    if os.path.isdir("/content/local_images"):
        paths["img_dir"] = "/content/local_images"

    print(
        "Image source:",
        paths["img_dir"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder = joblib.load(
        paths["meta_encoder"]
    )

    meta_dim = len(
        encoder.get_feature_names_out()
    )

    train_ds = MultimodalDermDataset(
        paths["train_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    test_ds = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    rows = []

    for seed in Config.SEEDS:

        checkpoint = os.path.join(
            paths["output_dir"],
            f"B6_PureCBM_seed_{seed}.pth"
        )

        model = MultimodalDermModel(
            num_classes=Config.NUM_CLASSES,
            num_concepts=Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="pure",
            use_metadata=True,
            meta_input_dim=meta_dim,
        ).to(device)

        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device
            ),
            strict=True,
        )

        (
            y_train,
            gt_train,
            soft_train,
        ) = extract(
            model,
            train_loader,
            device
        )

        (
            y_test,
            gt_test,
            soft_test,
        ) = extract(
            model,
            test_loader,
            device
        )

        hard_train = (
            soft_train >= 0.5
        ).astype(np.float32)

        hard_test = (
            soft_test >= 0.5
        ).astype(np.float32)

        representations = {
            "Soft predicted":
                (
                    soft_train,
                    soft_test
                ),

            "Hard predicted":
                (
                    hard_train,
                    hard_test
                ),

            "Ground truth":
                (
                    gt_train.astype(
                        np.float32
                    ),
                    gt_test.astype(
                        np.float32
                    )
                ),
        }

        for name, (
            x_train,
            x_test
        ) in representations.items():

            acc, f1 = evaluate_lr(
                x_train,
                y_train,
                x_test,
                y_test,
                seed,
            )

            rows.append({
                "Seed": seed,
                "Representation": name,
                "Accuracy": acc,
                "Macro F1": f1,
            })

            print(
                f"seed={seed} | "
                f"{name:15} | "
                f"F1={f1:.4f}"
            )

    df = pd.DataFrame(rows)

    summary = []

    for name in [
        "Soft predicted",
        "Hard predicted",
        "Ground truth"
    ]:

        sub = df[
            df["Representation"] == name
        ]

        summary.append({
            "Representation": name,
            "Accuracy":
                mean_sd(
                    sub["Accuracy"]
                ),
            "Macro F1":
                mean_sd(
                    sub["Macro F1"]
                ),
        })

    summary_df = pd.DataFrame(
        summary
    )

    out_dir = Path(
        paths["results_dir"]
    ) / "contribution_2"

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        out_dir /
        "c2_soft_bottleneck_seed_results.csv",
        index=False
    )

    summary_df.to_csv(
        out_dir /
        "c2_soft_bottleneck_summary.csv",
        index=False
    )

    print(
        "\n=== C2 SOFT BOTTLENECK DIAGNOSTIC ==="
    )

    print(
        summary_df.to_markdown(
            index=False
        )
    )


if __name__ == "__main__":
    main()
