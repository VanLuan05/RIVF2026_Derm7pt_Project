import torch
import torch.nn as nn

# Cập nhật hàm loss, thêm tham số disease_weights và tăng alpha
def compute_multitask_loss(disease_logits, disease_labels, concept_logits, concept_labels, disease_weights=None, alpha=2.0):
    """
    alpha đã được tăng lên 2.0 để AI phải chú ý học 7 Khái niệm lâm sàng nhiều hơn.
    """
    # 1. Loss cho chẩn đoán bệnh (Có dùng Class Weights)
    if disease_weights is not None:
        criterion_disease = nn.CrossEntropyLoss(weight=disease_weights)
    else:
        criterion_disease = nn.CrossEntropyLoss()
        
    loss_disease = criterion_disease(disease_logits, disease_labels)
    
    # 2. Loss cho Khái niệm lâm sàng 
    pos_weight = torch.tensor([5.0] * 7).to(concept_logits.device)
    criterion_concept = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_concept = criterion_concept(concept_logits, concept_labels)
    
    # 3. Tính tổng Loss
    total_loss = loss_disease + (alpha * loss_concept)
    
    return total_loss, loss_disease, loss_concept

#------------------------------------------------
import torch.optim as optim
from tqdm import tqdm
import numpy as np 

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=5e-5): # Tăng epoch, giảm learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    model.to(device)
    
    # --- BẮT ĐẦU PHẦN TÍNH TRỌNG SỐ LỚP ---
    print("Đang tính toán Class Weights để cân bằng dữ liệu...")
    # Lấy toàn bộ nhãn bệnh trong tập train
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['label_disease'].numpy())
        
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    num_classes = len(class_counts)
    
    # Công thức chuẩn: Trọng số = Tổng số mẫu / (Số lớp * Số mẫu của lớp đó)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Chuyển thành Tensor và đẩy lên GPU
    disease_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Hoàn tất tính trọng số!")
    # --- KẾT THÚC PHẦN TÍNH TRỌNG SỐ ---

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            clinic_img = batch['clinic_img'].to(device)
            derm_img = batch['derm_img'].to(device)
            disease_labels = batch['label_disease'].to(device)
            concept_labels = batch['concept_labels'].to(device) 
            
            optimizer.zero_grad()
            disease_out, concept_out = model(clinic_img, derm_img, meta_features=None)
            
            # Đưa disease_weights vào hàm loss
            loss, l_dis, l_con = compute_multitask_loss(
                disease_out, disease_labels, concept_out, concept_labels, disease_weights=disease_weights
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Phần Validation giữ nguyên như cũ, chỉ cập nhật hàm loss gọi disease_weights ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clinic_img = batch['clinic_img'].to(device)
                derm_img = batch['derm_img'].to(device)
                disease_labels = batch['label_disease'].to(device)
                concept_labels = batch['concept_labels'].to(device)
                
                disease_out, concept_out = model(clinic_img, derm_img, meta_features=None)
                # Đưa disease_weights vào hàm loss
                loss, _, _ = compute_multitask_loss(
                    disease_out, disease_labels, concept_out, concept_labels, disease_weights=disease_weights
                )
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nKết thúc Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Đã lưu mô hình tốt nhất tại Epoch {epoch+1}")