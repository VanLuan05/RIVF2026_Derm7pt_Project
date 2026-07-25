import torch
import torch.nn as nn

# Hàm tính toán tổng tổn thất (Loss)
def compute_multitask_loss(disease_logits, disease_labels, concept_logits, concept_labels, alpha=0.5):
    """
    alpha: Trọng số cân bằng giữa việc học dự đoán Bệnh và học Khái niệm.
           Nếu alpha = 0.5, mô hình coi trọng 2 nhiệm vụ như nhau.
    """
    # 1. Loss cho chẩn đoán bệnh (CrossEntropy vì là bài toán phân loại nhiều lớp)
    criterion_disease = nn.CrossEntropyLoss()
    loss_disease = criterion_disease(disease_logits, disease_labels)
    
    # 2. Loss cho Khái niệm lâm sàng (BCEWithLogitsLoss vì các khái niệm có thể xuất hiện đồng thời)
    # Lưu ý: Các concept_labels phải được chuyển về dạng float
    criterion_concept = nn.BCEWithLogitsLoss()
    loss_concept = criterion_concept(concept_logits, concept_labels)
    
    # 3. Tính tổng Loss
    total_loss = loss_disease + (alpha * loss_concept)
    
    return total_loss, loss_disease, loss_concept
import torch.optim as optim
from tqdm import tqdm # Thư viện tạo thanh tiến trình đẹp mắt

# --- (Phần này sẽ được ráp với Dataset và DataLoader của Tuần 2) ---
# Giả định bạn đã có train_loader và val_loader
# model = MultimodalDermModel(...)
# ------------------------------------------------------------------

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    model.to(device)
    
    # Sử dụng AdamW optimizer (tốt hơn Adam chuẩn cho các mô hình có ResNet)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 1. GIAI ĐOẠN HUẤN LUYỆN (TRAIN)
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # Lấy dữ liệu và đẩy lên GPU/CPU
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            # (Phần metadata tạm thời bỏ qua để code đơn giản, sẽ tích hợp sau)
            
            disease_labels = batch['label_disease'].to(device)
            
            # (Giả định bạn đã có concept_labels dạng tensor 0/1 từ Dataset)
            # concept_labels = batch['concept_labels'].to(device) 
            
            # --- Tạm thời dùng dummy concept labels để code có thể chạy logic ---
            concept_labels = torch.zeros(clinic_img.size(0), 7).to(device) 
            
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Chạy qua mô hình
            disease_out, concept_out = model(clinic_img, derm_img, meta_features=None)
            
            # Tính Loss
            loss, l_dis, l_con = compute_multitask_loss(disease_out, disease_labels, concept_out, concept_labels)
            
            # Cập nhật trọng số (Backpropagation)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 2. GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION)
        # Tại đây, chúng ta đóng băng mô hình, tắt tính toán gradient để tiết kiệm bộ nhớ
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clinic_img = batch['clinic_img'].to(device)
                derm_img = batch['derm_img'].to(device)
                disease_labels = batch['label_disease'].to(device)
                concept_labels = torch.zeros(clinic_img.size(0), 7).to(device)
                
                disease_out, concept_out = model(clinic_img, derm_img, meta_features=None)
                loss, _, _ = compute_multitask_loss(disease_out, disease_labels, concept_out, concept_labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nKết thúc Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 3. LƯU CHECKPOINT MÔ HÌNH TỐT NHẤT
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Đã lưu mô hình tốt nhất tại Epoch {epoch+1}")