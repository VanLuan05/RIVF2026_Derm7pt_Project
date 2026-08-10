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


def sample_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.4f} ± {sample_sd(arr):.4f}"


def soft_oracle(concepts):
    return torch.where(
        concepts > 0.5,
        torch.full_like(concepts, Config.ORACLE_POS_PROB),
        torch.full_like(concepts, Config.ORACLE_NEG_PROB),
    )


def evaluate_intervention(model, data_loader, device):
    """
    Oracle concept substitution on the same trained model.

    Ground-truth dataset concepts are converted to 0.05/0.95 soft
    probabilities to reduce the distribution shock of replacing learned
    probabilities by exact binary values.

    This is an oracle analysis, NOT a prospective clinician study.
    """
    model.eval()

    labels_all = []
    pred_ai_all = []
    pred_oracle_all = []

    with torch.inference_mode():
        for batch in data_loader:
            clinic_img = batch["clinic_img"].to(device, non_blocking=True)
            derm_img = batch["derm_img"].to(device, non_blocking=True)
            meta = batch["metadata"].to(device, non_blocking=True)
            labels = batch["label_disease"].cpu().numpy()

            gt = batch["concept_labels"].to(
                device, non_blocking=True
            ).float()
            oracle_probs = soft_oracle(gt)

            logits_ai, _ = model(
                clinic_img,
                derm_img,
                meta_features=meta,
            )
            logits_oracle, _ = model(
                clinic_img,
                derm_img,
                meta_features=meta,
                intervention_probs=oracle_probs,
            )

            pred_ai_all.extend(
                torch.argmax(logits_ai, dim=1).cpu().numpy()
            )
            pred_oracle_all.extend(
                torch.argmax(logits_oracle, dim=1).cpu().numpy()
            )
            labels_all.extend(labels)

    class_labels = list(range(Config.NUM_CLASSES))

    f1_ai = f1_score(
        labels_all,
        pred_ai_all,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    f1_oracle = f1_score(
        labels_all,
        pred_oracle_all,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    acc_ai = accuracy_score(labels_all, pred_ai_all)
    acc_oracle = accuracy_score(labels_all, pred_oracle_all)

    return {
        "f1_ai": float(f1_ai),
        "f1_oracle": float(f1_oracle),
        "acc_ai": float(acc_ai),
        "acc_oracle": float(acc_oracle),
    }


def main():
    paths = Config.ensure_runtime_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(paths["meta_encoder"]):
        raise FileNotFoundError(
            f"Thiếu metadata encoder: {paths['meta_encoder']}"
        )

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
        pin_memory=torch.cuda.is_available(),
    )

    experiments = [
        {"name": "B6_PureCBM", "bottleneck": "pure"},
        {"name": "Proposed_Hybrid", "bottleneck": "hybrid"},
    ]

    expected = [
        os.path.join(
            paths["output_dir"],
            f"{exp['name']}_seed_{seed}.pth",
        )
        for exp in experiments
        for seed in Config.SEEDS
    ]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu checkpoint cho oracle intervention:\n- "
            + "\n- ".join(missing)
        )

    per_seed_rows = []
    summary_rows = []

    for exp in experiments:
        f1_ai_list, f1_oracle_list = [], []
        acc_ai_list, acc_oracle_list = [], []

        for seed in Config.SEEDS:
            model_path = os.path.join(
                paths["output_dir"],
                f"{exp['name']}_seed_{seed}.pth",
            )

            model = MultimodalDermModel(
                num_classes=Config.NUM_CLASSES,
                num_concepts=Config.NUM_CONCEPTS,
                modality="dual",
                bottleneck_type=exp["bottleneck"],
                use_metadata=True,
                meta_input_dim=meta_input_dim,
            ).to(device)

            model.load_state_dict(
                torch.load(model_path, map_location=device),
                strict=True,
            )

            res = evaluate_intervention(model, test_loader, device)

            f1_ai_list.append(res["f1_ai"])
            f1_oracle_list.append(res["f1_oracle"])
            acc_ai_list.append(res["acc_ai"])
            acc_oracle_list.append(res["acc_oracle"])

            per_seed_rows.append(
                {
                    "Model": exp["name"],
                    "Seed": seed,
                    "Accuracy AI": res["acc_ai"],
                    "Accuracy Oracle": res["acc_oracle"],
                    "Delta Accuracy": res["acc_oracle"] - res["acc_ai"],
                    "Macro F1 AI": res["f1_ai"],
                    "Macro F1 Oracle": res["f1_oracle"],
                    "Delta Macro F1": res["f1_oracle"] - res["f1_ai"],
                }
            )

            print(
                f"{exp['name']} seed={seed}: "
                f"F1 AI={res['f1_ai']:.4f}, "
                f"F1 oracle={res['f1_oracle']:.4f}, "
                f"delta={res['f1_oracle'] - res['f1_ai']:+.4f}"
            )

        summary_rows.append(
            {
                "Model": exp["name"],
                "Accuracy AI": mean_sd(acc_ai_list),
                "Accuracy Oracle": mean_sd(acc_oracle_list),
                "Macro F1 AI": mean_sd(f1_ai_list),
                "Macro F1 Oracle": mean_sd(f1_oracle_list),
                "Mean Delta Macro F1": (
                    f"{np.mean(np.asarray(f1_oracle_list) - np.asarray(f1_ai_list)):+.4f}"
                ),
            }
        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = pd.DataFrame(summary_rows)

    out_path = os.path.join(
        paths["results_dir"], "intervention_results.md"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Oracle Concept Intervention Analysis\n\n")
        f.write(
            "Ground-truth Derm7pt concept labels are substituted as soft "
            f"oracle probabilities ({Config.ORACLE_NEG_PROB:.2f}/"
            f"{Config.ORACLE_POS_PROB:.2f}). This diagnoses concept "
            "dependence and must not be described as a real doctor "
            "intervention study.\n\n"
        )
        f.write("## Per-seed results\n\n")
        f.write(per_seed_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

    print("\n" + summary_df.to_markdown(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()