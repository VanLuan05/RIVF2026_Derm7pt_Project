import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.config import Config


def compute_multitask_loss(
    disease_logits,
    disease_labels,
    concept_logits,
    concept_labels,
    disease_weights=None,
    concept_pos_weights=None,
    alpha=1.0,
):
    """Loss chung cho baseline, Pure CBM và Hybrid CBM."""
    criterion_disease = nn.CrossEntropyLoss(weight=disease_weights)
    loss_disease = criterion_disease(disease_logits, disease_labels)

    if concept_logits is None:
        return loss_disease, loss_disease, loss_disease.new_zeros(())

    criterion_concept = nn.BCEWithLogitsLoss(pos_weight=concept_pos_weights)
    loss_concept = criterion_concept(concept_logits, concept_labels.float())
    total_loss = loss_disease + alpha * loss_concept
    return total_loss, loss_disease, loss_concept


def _evaluate_validation(
    model,
    val_loader,
    device,
    disease_weights=None,
    concept_pos_weights=None,
    alpha=1.0,
):
    model.eval()
    val_loss_sum = 0.0
    disease_true, disease_pred = [], []
    concept_true, concept_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            clinic_img = batch["clinic_img"].to(device)
            derm_img = batch["derm_img"].to(device)
            meta_features = batch["metadata"].to(device)
            disease_labels = batch["label_disease"].to(device)
            concept_labels = batch["concept_labels"].to(device).float()

            disease_out, concept_out = model(
                clinic_img,
                derm_img,
                meta_features=meta_features,
            )

            loss, _, _ = compute_multitask_loss(
                disease_out,
                disease_labels,
                concept_out,
                concept_labels,
                disease_weights=disease_weights,
                concept_pos_weights=concept_pos_weights,
                alpha=alpha,
            )
            val_loss_sum += loss.item()

            disease_true.extend(disease_labels.cpu().numpy())
            disease_pred.extend(torch.argmax(disease_out, dim=1).cpu().numpy())

            if concept_out is not None:
                concept_true.extend(concept_labels.cpu().numpy())
                concept_pred.extend((torch.sigmoid(concept_out) >= 0.5).int().cpu().numpy())

    val_loss = val_loss_sum / max(len(val_loader), 1)
    disease_f1 = f1_score(
        disease_true,
        disease_pred,
        average="macro",
        zero_division=0,
    )

    if concept_true:
        concept_f1 = f1_score(
            np.asarray(concept_true),
            np.asarray(concept_pred),
            average="macro",
            zero_division=0,
        )
    else:
        concept_f1 = np.nan

    return val_loss, disease_f1, concept_f1


def train_model(
    model,
    train_loader,
    val_loader,
    disease_weights=None,
    concept_pos_weights=None,
    num_epochs=20,
    learning_rate=5e-5,
    weight_decay=1e-4,
    experiment_name="best_model",
    alpha=1.0,
    monitor="disease_f1",
    patience=5,
    min_delta=1e-4,
):
    """
    Huấn luyện thống nhất cho toàn bộ thí nghiệm.

    Paper protocol:
    - AdamW cho mọi mô hình.
    - Class-weighted CE và concept pos_weight (nếu có).
    - Alpha được truyền tường minh.
    - Checkpoint được chọn trên Validation Disease Macro-F1 mặc định.
    - Early stopping chỉ nhìn Validation, không dùng Test.
    """
    if monitor not in {"disease_f1", "val_loss"}:
        raise ValueError("monitor phải là 'disease_f1' hoặc 'val_loss'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if disease_weights is not None:
        disease_weights = disease_weights.to(device)
    if concept_pos_weights is not None:
        concept_pos_weights = concept_pos_weights.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_disease_f1 = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    epochs_without_improvement = 0

    save_path = Config.get_checkpoint_path(experiment_name)
    metadata_path = os.path.splitext(save_path)[0] + "_training.json"

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        loop = tqdm(train_loader, leave=False)
        for batch in loop:
            clinic_img = batch["clinic_img"].to(device)
            derm_img = batch["derm_img"].to(device)
            meta_features = batch["metadata"].to(device)
            disease_labels = batch["label_disease"].to(device)
            concept_labels = batch["concept_labels"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            disease_out, concept_out = model(
                clinic_img,
                derm_img,
                meta_features=meta_features,
            )

            loss, _, _ = compute_multitask_loss(
                disease_out,
                disease_labels,
                concept_out,
                concept_labels,
                disease_weights=disease_weights,
                concept_pos_weights=concept_pos_weights,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            loop.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss, val_disease_f1, val_concept_f1 = _evaluate_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            disease_weights=disease_weights,
            concept_pos_weights=concept_pos_weights,
            alpha=alpha,
        )

        concept_text = "N/A" if np.isnan(val_concept_f1) else f"{val_concept_f1:.4f}"
        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Disease Macro-F1={val_disease_f1:.4f} | "
            f"Val Concept Macro-F1={concept_text}"
        )

        if monitor == "disease_f1":
            improved = val_disease_f1 > (best_disease_f1 + min_delta)
        else:
            improved = val_loss < (best_val_loss - min_delta)

        if improved:
            best_disease_f1 = val_disease_f1
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(model.state_dict(), save_path)
            metadata = {
                "experiment_name": experiment_name,
                "best_epoch": best_epoch,
                "monitor": monitor,
                "best_val_disease_macro_f1": float(best_disease_f1),
                "best_val_loss": float(best_val_loss),
                "val_concept_macro_f1_at_best_epoch": (
                    None if np.isnan(val_concept_f1) else float(val_concept_f1)
                ),
                "alpha": float(alpha),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "num_epochs_max": int(num_epochs),
                "patience": int(patience),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"  [+] Saved best checkpoint: {save_path}")
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(
                f"  [-] Early stopping tại epoch {epoch + 1}; "
                f"không cải thiện {monitor} trong {patience} epoch liên tiếp."
            )
            break

    if best_epoch < 0:
        raise RuntimeError("Không có checkpoint nào được lưu. Kiểm tra dữ liệu/metric validation.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)