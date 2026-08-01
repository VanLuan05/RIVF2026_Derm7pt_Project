import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

# 1. TẠO LỚP VỎ BỌC ĐỂ GRAD-CAM HIỂU ĐƯỢC MÔ HÌNH ĐA PHƯƠNG THỨC
class DermCamWrapper(torch.nn.Module):
    def __init__(self, model, clinic_img, meta_features, target_type='disease'):
        super(DermCamWrapper, self).__init__()
        self.model = model
        self.clinic_img = clinic_img
        self.meta_features = meta_features
        self.target_type = target_type

    def forward(self, derm_img):
        # Cố định clinic_img và meta_features, chỉ cho phép derm_img thay đổi để tính đạo hàm
        d_logits, c_logits = self.model(self.clinic_img, derm_img, self.meta_features)
        
        if self.target_type == 'disease':
            return d_logits
        else:
            return c_logits

# 2. HÀM GIẢI CHUẨN HÓA ĐỂ VẼ ẢNH MÀU GỐC
def denormalize_image(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    model_name = "best_model_P2"
    model_path = Config.get_checkpoint_path(experiment_name=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải dữ liệu Mapping
    with open(Config.LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
    idx_to_disease = {v: k for k, v in disease_to_idx.items()}

    # Khởi tạo DataLoader (Chỉ lấy batch size = 1 để dễ soi từng ảnh)
    dataset = MultimodalDermDataset(
        csv_file=Config.VAL_CSV, img_dir=Config.IMG_DIR, 
        label_mapping_path=Config.LABEL_MAPPING, transform=test_transforms
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Khởi tạo Mô hình Master P2
    model = MultimodalDermModel(
        num_classes=len(disease_to_idx), 
        num_concepts=7, 
        modality='dual',
        bottleneck_type='multitask',
        use_metadata=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Lấy 1 mẫu ngẫu nhiên từ Dataloader
    batch = next(iter(dataloader))
    clinic_img = batch['clinic_img'].to(device)
    derm_img = batch['derm_img'].to(device)
    meta_features = batch['metadata'].to(device)
    
    true_disease = idx_to_disease[batch['label_disease'].item()]

    # 3. CẤU HÌNH GRAD-CAM CHO BỆNH (DISEASE)
    # Lớp mục tiêu: Layer 4 của ResNet50 (layer tích chập cuối cùng trước khi pooling)
    target_layers = [model.derm_backbone[7]] 
    
    # Bọc mô hình lại và nhắm mục tiêu phân tích Bệnh
    disease_wrapper = DermCamWrapper(model, clinic_img, meta_features, target_type='disease')
    cam_disease = GradCAM(model=disease_wrapper, target_layers=target_layers)
    
    # Tính toán Heatmap
    grayscale_cam_dis = cam_disease(input_tensor=derm_img, targets=None)[0, :]
    
    # 4. CHUẨN BỊ ẢNH GỐC ĐỂ PHỦ NHIỆT (OVERLAY)
    rgb_img = denormalize_image(derm_img[0])
    visualization_dis = show_cam_on_image(rgb_img, grayscale_cam_dis, use_rgb=True)

    # 5. VẼ VÀ LƯU KẾT QUẢ
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Ảnh Gốc (Thực tế: {true_disease})")
    axes[0].axis('off')

    axes[1].imshow(visualization_dis)
    axes[1].set_title("Grad-CAM: Phân tích Bệnh")
    axes[1].axis('off')

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(Config.OUTPUT_DIR, f"gradcam_sample.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Đã lưu bản đồ nhiệt Grad-CAM tại: {save_path}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()