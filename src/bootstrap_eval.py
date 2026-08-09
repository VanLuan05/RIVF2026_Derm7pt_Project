import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def stratified_bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=2000, confidence=0.95, seed=42):
    """Bootstrap phân tầng theo lớp để mọi bootstrap sample giữ đủ lớp."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_indices = {c: np.where(y_true == c)[0] for c in classes}

    scores = []
    for _ in range(n_bootstraps):
        sampled_parts = []
        for c in classes:
            idx = class_indices[c]
            sampled_parts.append(rng.choice(idx, size=len(idx), replace=True))
        sample_idx = np.concatenate(sampled_parts)
        rng.shuffle(sample_idx)
        scores.append(metric_func(y_true[sample_idx], y_pred[sample_idx]))

    scores = np.asarray(scores, dtype=float)
    alpha = 1.0 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    point = metric_func(y_true, y_pred)
    return float(point), float(lower), float(upper)


def predict_probabilities(model, loader, device):
    y_true, probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            c_img = batch["clinic_img"].to(device)
            d_img = batch["derm_img"].to(device)
            meta = batch["metadata"].to(device)
            logits, _ = model(c_img, d_img, meta_features=meta)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.extend(batch["label_disease"].cpu().numpy())
    return np.asarray(y_true), np.vstack(probs)


def main():
    paths = Config.runtime_paths()
    os.makedirs(paths["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=workers)

    rows = []
    all_seed_probs = []
    reference_y = None

    for seed in Config.SEEDS:
        model_path = os.path.join(paths["output_dir"], f"Proposed_Hybrid_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"[skip] Missing {model_path}")
            continue

        model = MultimodalDermModel(
            num_classes=Config.NUM_CLASSES,
            num_concepts=Config.NUM_CONCEPTS,
            modality="dual",
            bottleneck_type="hybrid",
            use_metadata=True,
            meta_input_dim=meta_input_dim,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

        y_true, probs = predict_probabilities(model, test_loader, device)
        preds = probs.argmax(axis=1)
        reference_y = y_true if reference_y is None else reference_y
        all_seed_probs.append(probs)

        f1_point, f1_low, f1_high = stratified_bootstrap_ci(
            y_true,
            preds,
            lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        )
        acc_point, acc_low, acc_high = stratified_bootstrap_ci(
            y_true,
            preds,
            accuracy_score,
        )
        rows.append({
            "Model": f"Proposed_Hybrid seed {seed}",
            "Accuracy": f"{acc_point:.4f}",
            "Accuracy 95% CI": f"[{acc_low:.4f}, {acc_high:.4f}]",
            "Macro F1": f"{f1_point:.4f}",
            "Macro F1 95% CI": f"[{f1_low:.4f}, {f1_high:.4f}]",
        })

    if not all_seed_probs:
        raise RuntimeError("Không tìm thấy checkpoint Proposed_Hybrid.")

    # Bổ sung ensemble 3-seed như một phân tích phụ; không thay thế Mean ± SD của paper chính.
    ensemble_probs = np.mean(np.stack(all_seed_probs, axis=0), axis=0)
    ensemble_pred = ensemble_probs.argmax(axis=1)
    f1_point, f1_low, f1_high = stratified_bootstrap_ci(
        reference_y,
        ensemble_pred,
        lambda a, b: f1_score(a, b, average="macro", zero_division=0),
    )
    acc_point, acc_low, acc_high = stratified_bootstrap_ci(
        reference_y,
        ensemble_pred,
        accuracy_score,
    )
    rows.append({
        "Model": "Proposed_Hybrid 3-seed probability ensemble (secondary)",
        "Accuracy": f"{acc_point:.4f}",
        "Accuracy 95% CI": f"[{acc_low:.4f}, {acc_high:.4f}]",
        "Macro F1": f"{f1_point:.4f}",
        "Macro F1 95% CI": f"[{f1_low:.4f}, {f1_high:.4f}]",
    })

    df = pd.DataFrame(rows)
    out_path = os.path.join(paths["results_dir"], "bootstrap_ci.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Stratified Bootstrap 95% Confidence Intervals\n\n")
        f.write(
            "Bootstrap resampling is stratified by disease class. The main paper should still report "
            "mean ± sample SD across independent training seeds; the ensemble row is secondary analysis only.\n\n"
        )
        f.write(df.to_markdown(index=False))

    print(df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()