import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
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


def evaluate_intervention(model, data_loader, device):
    """
    Oracle intervention analysis.
    Ground-truth dataset concepts are substituted for predicted concept probabilities.
    This is NOT a prospective doctor study.
    """
    model.eval()
    labels_all, pred_ai_all, pred_oracle_all = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            clinic_img = batch["clinic_img"].to(device)
            derm_img = batch["derm_img"].to(device)
            meta = batch["metadata"].to(device)
            labels = batch["label_disease"].cpu().numpy()

            # Soft oracle intervention reduces hard 0/1 distribution shock.
            gt = batch["concept_labels"].to(device).float()
            oracle_probs = torch.where(
                gt > 0.5,
                torch.full_like(gt, 0.95),
                torch.full_like(gt, 0.05),
            )

            logits_ai, _ = model(clinic_img, derm_img, meta_features=meta)
            logits_oracle, _ = model(
                clinic_img,
                derm_img,
                meta_features=meta,
                intervention_probs=oracle_probs,
            )

            pred_ai_all.extend(torch.argmax(logits_ai, dim=1).cpu().numpy())
            pred_oracle_all.extend(torch.argmax(logits_oracle, dim=1).cpu().numpy())
            labels_all.extend(labels)

    f1_ai = f1_score(labels_all, pred_ai_all, average="macro", zero_division=0)
    f1_oracle = f1_score(labels_all, pred_oracle_all, average="macro", zero_division=0)
    return f1_ai, f1_oracle


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

    experiments = [
        {"name": "B6_PureCBM", "bottleneck": "pure"},
        {"name": "Proposed_Hybrid", "bottleneck": "hybrid"},
    ]

    rows = []
    for exp in experiments:
        f1_ai_list, f1_oracle_list = [], []
        for seed in Config.SEEDS:
            model_path = os.path.join(paths["output_dir"], f"{exp['name']}_seed_{seed}.pth")
            if not os.path.exists(model_path):
                print(f"[skip] Missing {model_path}")
                continue

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type=exp["bottleneck"],
                use_metadata=True,
                meta_input_dim=meta_input_dim,
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

            f1_ai, f1_oracle = evaluate_intervention(model, test_loader, device)
            f1_ai_list.append(f1_ai)
            f1_oracle_list.append(f1_oracle)
            print(
                f"{exp['name']} seed={seed}: AI={f1_ai:.4f}, "
                f"oracle={f1_oracle:.4f}, delta={f1_oracle - f1_ai:+.4f}"
            )

        if f1_ai_list:
            delta = np.mean(f1_oracle_list) - np.mean(f1_ai_list)
            rows.append({
                "Model": exp["name"],
                "Macro F1 (AI concepts)": mean_sd(f1_ai_list),
                "Macro F1 (Oracle concepts)": mean_sd(f1_oracle_list),
                "Delta": f"{delta:+.4f}",
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(paths["results_dir"], "intervention_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Oracle Concept Intervention Analysis\n\n")
        f.write(
            "Ground-truth Derm7pt concept labels are used as an oracle substitute for predicted concepts. "
            "This analysis diagnoses concept dependence; it must not be described as a real doctor intervention study.\n\n"
        )
        f.write(df.to_markdown(index=False))

    print("\n" + df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()