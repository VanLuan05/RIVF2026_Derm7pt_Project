import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy kiểm định trên thiết bị: {device}")

    # 1. Khởi tạo dữ liệu Test
    with open(Config.LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
        
    test_dataset = MultimodalDermDataset(
        csv_file=Config.TEST_CSV, # Bắt buộc là tập TEST
        img_dir=Config.IMG_DIR, 
        label_mapping_path=Config.LABEL_MAPPING, 
        transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    num_classes = len(disease_to_idx)

    # 2. Danh sách các seed đã huấn luyện
    seeds = [42, 100, 2026]
    acc_results = []
    f1_results = []

    print("\n" + "="*60)
    print(" BẮT ĐẦU KIỂM THỬ ĐỘ ỔN ĐỊNH (ROBUSTNESS TEST)")
    print("="*60)

    for seed in seeds:
        model_name = f"best_model_P2_seed_{seed}"
        model_path = Config.get_checkpoint_path(experiment_name=model_name)
        
        if not os.path.exists(model_path):
            print(f"Không tìm thấy file: {model_path}")
            continue

        # Khởi tạo lại mô hình và nạp trọng số
        model = MultimodalDermModel(
            num_classes=num_classes, 
            num_concepts=7, 
            modality='dual',
            bottleneck_type='multitask',
            use_metadata=True
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_labels = []
        all_preds = []

        # Chạy suy luận
        with torch.no_grad():
            for batch in test_loader:
                c_img = batch['clinic_img'].to(device)
                d_img = batch['derm_img'].to(device)
                m = batch['metadata'].to(device)
                labels = batch['label_disease'].numpy()
                
                logits, _ = model(c_img, d_img, meta_features=m)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
                
                all_labels.extend(labels)
                all_preds.extend(preds)

        # Tính toán điểm số cho seed hiện tại
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        acc_results.append(acc)
        f1_results.append(f1)
        
        print(f"Seed {seed:<4} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    # 3. Tổng hợp báo cáo Thống kê
    if len(acc_results) == 3:
        acc_mean, acc_std = np.mean(acc_results), np.std(acc_results)
        f1_mean, f1_std = np.mean(f1_results), np.std(f1_results)

        print("\n" + "="*60)
        print(" KẾT QUẢ THỐNG KÊ CUỐI CÙNG (REPORT READY)")
        print("="*60)
        print(f"Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"F1-Score : {f1_mean:.4f} ± {f1_std:.4f}")
        print("="*60)
        print("Bạn hãy copy đúng 2 dòng cuối này dán vào báo cáo nhé!")

if __name__ == "__main__":
    main()