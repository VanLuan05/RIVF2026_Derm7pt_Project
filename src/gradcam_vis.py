import json
import os
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def inverse_normalize(
    tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Return an RGB float image in [0, 1] without modifying input."""
    img = tensor.detach().clone().cpu()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy().astype(np.float32)


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")


class ClinicCamWrapper(torch.nn.Module):
    def __init__(self, model, derm_img, meta_features):
        super().__init__()
        self.model = model
        self.derm_img = derm_img
        self.meta_features = meta_features

    def forward(self, clinic_img):
        logits, _ = self.model(
            clinic_img,
            self.derm_img,
            meta_features=self.meta_features,
        )
        return logits


class DermCamWrapper(torch.nn.Module):
    def __init__(self, model, clinic_img, meta_features):
        super().__init__()
        self.model = model
        self.clinic_img = clinic_img
        self.meta_features = meta_features

    def forward(self, derm_img):
        logits, _ = self.model(
            self.clinic_img,
            derm_img,
            meta_features=self.meta_features,
        )
        return logits


def main():
    paths = Config.ensure_runtime_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(
        paths["output_dir"], "gradcam_results"
    )
    os.makedirs(output_dir, exist_ok=True)

    required = [
        paths["test_csv"],
        paths["label_mapping"],
        paths["meta_encoder"],
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Thiếu file Grad-CAM:\n- " + "\n- ".join(missing)
        )

    with open(paths["label_mapping"], "r", encoding="utf-8") as f:
        disease_to_idx = json.load(f)
    idx_to_disease = {int(v): k for k, v in disease_to_idx.items()}

    encoder = joblib.load(paths["meta_encoder"])
    meta_input_dim = len(encoder.get_feature_names_out())

    dataset = MultimodalDermDataset(
        paths["test_csv"],
        paths["img_dir"],
        paths["label_mapping"],
        meta_encoder_path=paths["meta_encoder"],
        transform=test_transforms,
    )

    # Deterministic order: Grad-CAM examples are not selected by Test score.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    seed = Config.GRADCAM_SEED
    model_path = os.path.join(
        paths["output_dir"],
        f"Proposed_Hybrid_seed_{seed}.pth",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy {model_path}. "
            "Hãy train đủ 21 models trước."
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
    model.eval()

    # Aim for one success and one failure example per true class.
    saved = {
        "Success": set(),
        "Failure": set(),
    }

    print(
        f"Generating Grad-CAM from pre-specified "
        f"Proposed_Hybrid seed={seed}"
    )

    for i, batch in enumerate(loader):
        if (
            len(saved["Success"]) >= Config.NUM_CLASSES
            and len(saved["Failure"]) >= Config.NUM_CLASSES
        ):
            break

        c_img = batch["clinic_img"].to(device)
        d_img = batch["derm_img"].to(device)
        meta = batch["metadata"].to(device)
        true_idx = int(batch["label_disease"].item())

        with torch.inference_mode():
            logits, _ = model(
                c_img,
                d_img,
                meta_features=meta,
            )
            probs = F.softmax(logits, dim=1)
            confidence_t, pred_idx_t = torch.max(probs, dim=1)

        pred_idx = int(pred_idx_t.item())
        confidence = float(confidence_t.item())

        status = "Success" if pred_idx == true_idx else "Failure"

        # Keep at most one example per true class per status.
        if true_idx in saved[status]:
            continue

        target = [ClassifierOutputTarget(pred_idx)]

        clinic_wrapper = ClinicCamWrapper(model, d_img, meta)
        derm_wrapper = DermCamWrapper(model, c_img, meta)

        clinic_target_layer = model.clinic_backbone[-2][-1]
        derm_target_layer = model.derm_backbone[-2][-1]

        with GradCAM(
            model=clinic_wrapper,
            target_layers=[clinic_target_layer],
        ) as cam:
            grayscale_clinic = cam(
                input_tensor=c_img,
                targets=target,
            )[0]

        with GradCAM(
            model=derm_wrapper,
            target_layers=[derm_target_layer],
        ) as cam:
            grayscale_derm = cam(
                input_tensor=d_img,
                targets=target,
            )[0]

        rgb_c = inverse_normalize(c_img[0])
        rgb_d = inverse_normalize(d_img[0])

        cam_c = show_cam_on_image(
            rgb_c, grayscale_clinic, use_rgb=True
        )
        cam_d = show_cam_on_image(
            rgb_d, grayscale_derm, use_rgb=True
        )

        true_name = idx_to_disease[true_idx]
        pred_name = idx_to_disease[pred_idx]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"[{status}] True: {true_name} | "
            f"Pred: {pred_name} | "
            f"Confidence: {100 * confidence:.1f}%",
            fontsize=13,
            fontweight="bold",
        )

        axes[0].imshow(rgb_c)
        axes[0].set_title("Clinical Image")
        axes[0].axis("off")

        axes[1].imshow(cam_c)
        axes[1].set_title("Clinical Grad-CAM")
        axes[1].axis("off")

        axes[2].imshow(rgb_d)
        axes[2].set_title("Dermoscopy Image")
        axes[2].axis("off")

        axes[3].imshow(cam_d)
        axes[3].set_title("Dermoscopy Grad-CAM")
        axes[3].axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.90))

        save_path = os.path.join(
            output_dir,
            (
                f"{status}_true-{safe_name(true_name)}_"
                f"pred-{safe_name(pred_name)}_idx-{i}.png"
            ),
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved[status].add(true_idx)
        print(f"Saved: {save_path}")

    print(
        "\nGrad-CAM complete. "
        f"Success classes represented={len(saved['Success'])}/"
        f"{Config.NUM_CLASSES}; "
        f"Failure classes represented={len(saved['Failure'])}/"
        f"{Config.NUM_CLASSES}."
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()