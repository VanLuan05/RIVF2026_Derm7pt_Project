
import os
import json
import joblib
from datetime import datetime
from pathlib import Path

from src.config import Config
from src.data.dataset import calculate_dataset_weights
from src.run_ablation import run_experiment


def main():
    paths = Config.ensure_runtime_dirs()

    # Prefer local Colab image cache for faster training
    local_img_dir = "/content/local_images"

    if os.path.isdir(local_img_dir):
        paths["img_dir"] = local_img_dir
        print(f"✅ C2 using LOCAL images: {paths['img_dir']}")
    else:
        print(f"⚠️ Local images not found. Using: {paths['img_dir']}")

    required_files = [
        paths["train_csv"],
        paths["val_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
        paths["selected_alpha"],
    ]

    missing = [p for p in required_files if not os.path.exists(p)]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n- " + "\n- ".join(missing)
        )

    # 1. Alpha đã tuning trên Validation
    selected_alpha = Config.load_selected_alpha()

    # 2. Metadata input dimension
    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

    # 3. Tính lại weights từ concept labels mới
    disease_weights, concept_pos_weights = calculate_dataset_weights(
        paths["train_csv"],
        paths["label_mapping"],
    )

    # 4. Chỉ train 2 architecture thuộc C2
    target_names = {
        "B6_PureCBM",
        "Proposed_Hybrid",
    }

    experiments = [
        exp
        for exp in Config.PAPER_EXPERIMENTS
        if exp["name"] in target_names
    ]

    found_names = {exp["name"] for exp in experiments}

    if found_names != target_names:
        raise RuntimeError(
            f"C2 architectures mismatch. Found: {sorted(found_names)}"
        )

    expected_runs = len(experiments) * len(Config.SEEDS)

    if expected_runs != 6:
        raise RuntimeError(
            f"Expected 6 C2 runs, got {expected_runs}."
        )

    print("\n" + "#" * 76)
    print("C2 RETRAIN PROTOCOL")
    print(f"Architectures       : {[e['name'] for e in experiments]}")
    print(f"Seeds               : {list(Config.SEEDS)}")
    print(f"Selected alpha      : {selected_alpha}")
    print(f"Metadata dimension  : {meta_input_dim}")
    print(f"Concept pos weights : {concept_pos_weights.tolist()}")
    print(f"Expected runs       : {expected_runs}")
    print("#" * 76)

    training_records = []

    for exp in experiments:
        for seed in Config.SEEDS:
            record = run_experiment(
                exp=exp,
                seed=seed,
                paths=paths,
                meta_input_dim=meta_input_dim,
                disease_weights=disease_weights,
                concept_pos_weights=concept_pos_weights,
                selected_alpha=selected_alpha,
            )

            training_records.append(record)

    if len(training_records) != 6:
        raise RuntimeError(
            f"C2 retrain finished with "
            f"{len(training_records)}/6 runs."
        )

    # 5. Save manifest riêng cho C2
    out_dir = Path(paths["results_dir"]) / "contribution_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "contribution": "C2 - Concept Bottleneck Models",
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "concept_encoding": {
            "num_concepts": Config.NUM_CONCEPTS,
            "status": "corrected Derm7pt binary encoding",
            "regression_structures_positive": [
                "blue areas",
                "white areas",
                "combinations",
            ],
            "vascular_structures_positive": [
                "dotted",
                "linear irregular",
            ],
        },

        "selected_alpha": float(selected_alpha),
        "seeds": list(Config.SEEDS),

        "architectures": [
            "B6_PureCBM",
            "Proposed_Hybrid",
        ],

        "num_completed_models": len(training_records),

        "optimizer": "AdamW",
        "learning_rate": Config.LEARNING_RATE,
        "weight_decay": Config.WEIGHT_DECAY,
        "max_epochs": Config.NUM_EPOCHS,
        "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE,
        "checkpoint_monitor": "validation_disease_macro_f1",

        "concept_pos_weights": [
            float(x)
            for x in concept_pos_weights.tolist()
        ],

        "experiments": training_records,
    }

    manifest_path = out_dir / "c2_training_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 76)
    print("C2 RETRAIN COMPLETED")
    print("Models trained :", len(training_records))
    print("Manifest       :", manifest_path)
    print("=" * 76)


if __name__ == "__main__":
    main()
