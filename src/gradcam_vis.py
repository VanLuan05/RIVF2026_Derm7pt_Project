import os
import json
import warnings
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

# Chặn cảnh báo rác của scikit-learn để log sạch sẽ
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Đưa ảnh từ Tensor về lại định dạng RGB để hiển thị"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).cpu().numpy()

# =====================================================================
# LỚP VỎ BỌC (WRAPPER) ĐỂ XỬ LÝ LỖI MULTIMODAL CHO THƯ VIỆN GRAD-CAM
# =====================================================================
class ClinicCamWrapper(torch.nn.Module):
    def __init__(self, model, derm_img, meta_features):
        super().__init__()
        self.model = model
        self.derm_img = derm_img
        self.meta_features = meta_features
    def forward(self, clinic_img):
        logits, _ = self.model(clinic_img, self.derm_img, meta_features=self.meta_features)
        return logits

class DermCamWrapper(torch.nn.Module):
    def __init__(self, model, clinic_img, meta_features):
        super().__init__()
        self.model = model
        self.clinic_img = clinic_img
        self.meta_features = meta_features
    def forward(self, derm_img):
        logits, _ = self.model(self.clinic_img, derm_img, meta_features=self.meta_features)
        return logits
# =====================================================================

def main():
    print("="*60)
    print("TRÍCH XUẤT GRAD-CAM (SUCCESS & FAILURE CASES)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đường dẫn và Dữ liệu
    base_dir = "data/"
    TEST_CSV = os.path.join(base_dir, "processed/test_split.csv")
    LABEL_MAPPING = os.path.join(base_dir, "processed/label_mapping.json")
    IMG_DIR = os.path.join(base_dir, "raw/images/")
    OUTPUT_DIR = "outputs/gradcam_results/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
    idx_to_disease = {v: k for k, v in disease_to_idx.items()}
    
    test_dataset = MultimodalDermDataset(TEST_CSV, IMG_DIR, LABEL_MAPPING, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
   # 2. Khởi tạo mô hình Proposed_Hybrid
    print("Đang nạp mô hình Proposed_Hybrid...")
    try:
        encoder = joblib.load(os.path.join(OUTPUT_DIR, "meta_encoder.joblib"))
        dynamic_meta_dim = len(encoder.get_feature_names_out())
    except:
        dynamic_meta_dim = 14
    model = MultimodalDermModel(num_classes=5, num_concepts=7, modality='dual', bottleneck_type='hybrid', use_metadata=True, meta_input_dim=dynamic_meta_dim).to(device)
    
    model_path = "outputs/Proposed_Hybrid_seed_42.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    saved_counts = {"Success": 0, "Failure": 0}
    max_per_category = 5 # Xuất 5 ca đúng, 5 ca sai để phân tích
    
    print("Đang quét tập Test để tạo ảnh Grad-CAM...")
    for i, batch in enumerate(test_loader):
        if saved_counts["Success"] >= max_per_category and saved_counts["Failure"] >= max_per_category:
            break
            
        c_img = batch['clinic_img'].to(device)
        d_img = batch['derm_img'].to(device)
        meta = batch['metadata'].to(device)
        true_label_idx = batch['label_disease'].item()
        true_label_name = idx_to_disease[true_label_idx]
        
        # Dự đoán để phân loại Đúng/Sai
        logits, _ = model(c_img, d_img, meta_features=meta)
        probs = F.softmax(logits, dim=1)
        pred_prob, pred_idx = torch.max(probs, dim=1)
        pred_label_name = idx_to_disease[pred_idx.item()]
        confidence = pred_prob.item() * 100
        
        status = "Success" if true_label_idx == pred_idx.item() else "Failure"
        if saved_counts[status] >= max_per_category:
            continue
            
        # 3. Tạo Wrapper và chạy Grad-CAM cho từng ảnh trong Batch
        clinic_model_wrapped = ClinicCamWrapper(model, d_img, meta)
        derm_model_wrapped = DermCamWrapper(model, c_img, meta)
        
        cam_clinic = GradCAM(model=clinic_model_wrapped, target_layers=[model.clinic_backbone[-2][-1]])
        cam_derm = GradCAM(model=derm_model_wrapped, target_layers=[model.derm_backbone[-2][-1]])
        
        # Tính toán Heatmap
        grayscale_cam_clinic = cam_clinic(input_tensor=c_img, targets=None)[0, :]
        grayscale_cam_derm = cam_derm(input_tensor=d_img, targets=None)[0, :]
        
        # Giải phóng bộ nhớ CAM ngay sau khi dùng
        del cam_clinic, cam_derm 
        
        # Chuyển ảnh gốc về RGB numpy
        rgb_c_img = inverse_normalize(c_img[0])
        rgb_d_img = inverse_normalize(d_img[0])
        
        # Áp mask màu lên ảnh
        cam_image_clinic = show_cam_on_image(rgb_c_img, grayscale_cam_clinic, use_rgb=True)
        cam_image_derm = show_cam_on_image(rgb_d_img, grayscale_cam_derm, use_rgb=True)
        
        # Vẽ biểu đồ kết hợp
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"[{status}] True: {true_label_name} | Pred: {pred_label_name} (Conf: {confidence:.1f}%)", fontsize=14, fontweight='bold', color='green' if status=="Success" else 'red')
        
        axes[0].imshow(rgb_c_img); axes[0].set_title("Clinic Image"); axes[0].axis('off')
        axes[1].imshow(cam_image_clinic); axes[1].set_title("Clinic Grad-CAM"); axes[1].axis('off')
        axes[2].imshow(rgb_d_img); axes[2].set_title("Dermoscopy Image"); axes[2].axis('off')
        axes[3].imshow(cam_image_derm); axes[3].set_title("Dermoscopy Grad-CAM"); axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{status}_{i}_True_{true_label_name}_Pred_{pred_label_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        saved_counts[status] += 1
        print(f"Đã lưu ảnh {status}: {save_path}")

    print("Hoàn tất xuất ảnh XAI! Vui lòng vào thư mục outputs/gradcam_results/ để xem.")

if __name__ == "__main__":
    main()