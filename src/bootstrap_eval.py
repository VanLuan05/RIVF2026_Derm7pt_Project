import json
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import MultimodalDermDataset, test_transforms
from src.models.models import MultimodalDermModel

def compute_bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, alpha=0.05):
    """Tính toán Khoảng tin cậy (Confidence Interval) bằng Bootstrapping"""
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        # Bốc mẫu ngẫu nhiên có hoàn lại
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        
        # Chỉ tính toán nếu mẫu bốc được chứa nhiều hơn 1 class (để tránh lỗi macro F1)
        if len(np.unique(y_true[sample_indices])) > 1:
            score = metric_func(y_true[sample_indices], y_pred[sample_indices])
            bootstrapped_scores.append(score)
            
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Cắt bỏ râu ria 2.5% hai đầu để lấy 95% độ tin cậy ở giữa
    lower_bound = np.percentile(sorted_scores, (alpha / 2) * 100)
    upper_bound = np.percentile(sorted_scores, (1 - alpha / 2) * 100)
    mean_score = np.mean(sorted_scores)
    
    return mean_score, lower_bound, upper_bound

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Gọi Master Model P2 ra trận
    model_name = "best_model_P2"
    model_path = Config.get_checkpoint_path(experiment_name=model_name)
    
    with open(Config.LABEL_MAPPING, 'r') as f:
        disease_to_idx = json.load(f)
        
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

    # 2. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
    print("Đang chạy suy luận trên tập TEST...")
    test_dataset = MultimodalDermDataset(
        csv_file=Config.TEST_CSV,  # <--- Bắt buộc dùng Test CSV
        img_dir=Config.IMG_DIR, 
        label_mapping_path=Config.LABEL_MAPPING, 
        transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            c_img = batch['clinic_img'].to(device)
            d_img = batch['derm_img'].to(device)
            m = batch['metadata'].to(device)
            labels = batch['label_disease'].numpy()
            
            d_logits, _ = model(c_img, d_img, meta_features=m)
            probs = torch.softmax(d_logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_labels.extend(labels)
            all_preds.extend(preds)
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 3. CHẠY BOOTSTRAPPING (1000 lần lấy mẫu)
    print("Đang tính toán Khoảng tin cậy 95% (1000 Bootstraps)...")
    def macro_f1(y_t, y_p):
        return f1_score(y_t, y_p, average='macro')
        
    f1_mean, f1_lower, f1_upper = compute_bootstrap_ci(y_true, y_pred, macro_f1)
    acc_mean, acc_lower, acc_upper = compute_bootstrap_ci(y_true, y_pred, accuracy_score)

    print("="*60)
    print("BÁO CÁO TÍNH BỀN VỮNG (ROBUSTNESS) TRÊN TẬP TEST")
    print("="*60)
    print(f"Accuracy (95% CI) : {acc_mean:.4f}  [{acc_lower:.4f} - {acc_upper:.4f}]")
    print(f"F1-Score (95% CI) : {f1_mean:.4f}  [{f1_lower:.4f} - {f1_upper:.4f}]")
    print("="*60)
    print("F1-Score = {:.2f} ± {:.2f}".format(f1_mean, (f1_upper - f1_lower)/2))

if __name__ == "__main__":
    main()