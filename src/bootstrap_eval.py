import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def stratified_bootstrap_indices(y_true, rng):
    """
    Resample with replacement inside each disease class, preserving the
    original class counts in every bootstrap replicate.
    """
    sampled_parts = []

    for cls in np.unique(y_true):
        cls_idx = np.flatnonzero(y_true == cls)
        sampled_parts.append(
            rng.choice(cls_idx, size=len(cls_idx), replace=True)
        )

    sampled = np.concatenate(sampled_parts)
    rng.shuffle(sampled)
    return sampled


def bootstrap_ci(
    y_true,
    y_pred,
    metric_func,
    n_bootstraps,
    random_state,
    alpha=0.05,
):
    rng = np.random.default_rng(random_state)
    scores = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        idx = stratified_bootstrap_indices(y_true, rng)
        scores[i] = metric_func(y_true[idx], y_pred[idx])

    point = float(metric_func(y_true, y_pred))
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return point, lower, upper


def collect_predictions(model, loader, device):
    y_true, probs_all = [], []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            c_img = batch["clinic_img"].to(device, non_blocking=True)
            d_img = batch["derm_img"].to(device, non_blocking=True)
            meta = batch["metadata"].to(device, non_blocking=True)

            logits, _ = model(c_img, d_img, meta_features=meta)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            y_true.extend(batch["label_disease"].cpu().numpy())
            probs_all.extend(probs)

    return np.asarray(y_true), np.asarray(probs_all)


def main():
    paths = Config.ensure_runtime_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

    dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    workers = 2 if paths["data_root"].startswith("/content/") else 0
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_paths = [
        os.path.join(
            paths["output_dir"],
            f"Proposed_Hybrid_seed_{seed}.pth",
        )
        for seed in Config.SEEDS
    ]
    missing = [p for p in checkpoint_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu Proposed Hybrid checkpoint:\n- "
            + "\n- ".join(missing)
        )

    class_labels = list(range(Config.NUM_CLASSES))

    metric_functions = {
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Macro F1": lambda yt, yp: f1_score(
            yt,
            yp,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ),
    }

    rows = []
    probability_runs = []
    reference_y_true = None

    for seed_idx, seed in enumerate(Config.SEEDS):
        model_path = os.path.join(
            paths["output_dir"],
            f"Proposed_Hybrid_seed_{seed}.pth",
        )

        model = MultimodalDermModel(
            num_classes=Config.NUM_CLASSES,
            num_concepts=Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="hybrid",
            use_metadata=True,
            meta_input_dim=meta_input_dim,
        ).to(device)

        model.load_state_dict(
            torch.load(model_path, map_location=device),
            strict=True,
        )

        y_true, probs = collect_predictions(model, loader, device)
        y_pred = np.argmax(probs, axis=1)

        if reference_y_true is None:
            reference_y_true = y_true
        elif not np.array_equal(reference_y_true, y_true):
            raise RuntimeError("Test sample order changed across seed runs.")

        probability_runs.append(probs)

        for metric_name, metric_func in metric_functions.items():
            point, low, high = bootstrap_ci(
                y_true,
                y_pred,
                metric_func,
                n_bootstraps=Config.BOOTSTRAP_ITERATIONS,
                random_state=(
                    Config.BOOTSTRAP_RANDOM_STATE
                    + 1000 * seed_idx
                ),
            )
            rows.append(
                {
                    "Analysis": f"Seed {seed}",
                    "Metric": metric_name,
                    "Point Estimate": point,
                    "95% CI Lower": low,
                    "95% CI Upper": high,
                }
            )

    # Secondary analysis: probability ensemble across the three independently
    # trained seeds. Do not confuse this with mean ± SD of independent runs.
    ensemble_probs = np.mean(np.stack(probability_runs, axis=0), axis=0)
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    for metric_name, metric_func in metric_functions.items():
        point, low, high = bootstrap_ci(
            reference_y_true,
            ensemble_pred,
            metric_func,
            n_bootstraps=Config.BOOTSTRAP_ITERATIONS,
            random_state=Config.BOOTSTRAP_RANDOM_STATE + 9999,
        )
        rows.append(
            {
                "Analysis": "3-seed probability ensemble (secondary)",
                "Metric": metric_name,
                "Point Estimate": point,
                "95% CI Lower": low,
                "95% CI Upper": high,
            }
        )

    df = pd.DataFrame(rows)

    out_path = os.path.join(paths["results_dir"], "bootstrap_ci.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Bootstrap Confidence Intervals — Proposed Hybrid\n\n")
        f.write(
            f"Stratified percentile bootstrap with "
            f"{Config.BOOTSTRAP_ITERATIONS} replicates. Resampling is "
            "performed within each disease class, so every replicate "
            "preserves the Test-set class counts.\n\n"
        )
        f.write(
            "The paper's primary across-training-run uncertainty remains "
            "mean ± sample SD over seeds. The probability ensemble is a "
            "secondary analysis and should be labeled as such.\n\n"
        )
        f.write(df.to_markdown(index=False, floatfmt=".4f"))

    print(df.to_markdown(index=False, floatfmt=".4f"))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()