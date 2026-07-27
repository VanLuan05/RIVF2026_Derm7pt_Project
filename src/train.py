import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def compute_multitask_loss(disease_logits, disease_labels, concept_logits, concept_labels, disease_weights=None, concept_pos_weights=None, alpha=2.0):
    # 1. Loss chẩn đoán bệnh (Cân bằng 5 lớp)
    if disease_weights is not None:
        criterion_disease = nn.CrossEntropyLoss(weight=disease_weights)
    else:
        criterion_disease = nn.CrossEntropyLoss()
        
    loss_disease = criterion_disease(disease_logits, disease_labels)
    
    # KHI CHẠY BASELINE (B0, B1, B2): Nếu không có Concept đầu ra, chỉ trả về Loss Bệnh
    if concept_logits is None:
        return loss_disease, loss_disease, torch.tensor(0.0)
    
    # 2. Loss khái niệm (Dành cho Pure CBM và Hybrid CBM)
    if concept_pos_weights is not None:
        criterion_concept = nn.BCEWithLogitsLoss(pos_weight=concept_pos_weights)
    else:
        criterion_concept = nn.BCEWithLogitsLoss()
        
    loss_concept = criterion_concept(concept_logits, concept_labels)
    
    # 3. Tính tổng Loss
    total_loss = loss_disease + (alpha * loss_concept)
    return total_loss, loss_disease, loss_concept

def train_model(model, train_loader, val_loader, disease_weights=None, concept_pos_weights=None, num_epochs=20, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    model.to(device)
    
    # Đẩy các ma trận trọng số lên GPU
    if disease_weights is not None:
        disease_weights = disease_weights.to(device)
    if concept_pos_weights is not None:
        concept_pos_weights = concept_pos_weights.to(device)

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
            
            # Đẩy Metadata lên GPU
            meta_features = batch['metadata'].to(device)
            
            optimizer.zero_grad()
            
            # [SỬA LỖI]: Truyền meta_features vào Model lúc Train
            disease_out, concept_out = model(clinic_img, derm_img, meta_features=meta_features)
            
            loss, l_dis, l_con = compute_multitask_loss(
                disease_out, disease_labels, concept_out, concept_labels, 
                disease_weights=disease_weights, concept_pos_weights=concept_pos_weights
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Chuyển sang chế độ Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                clinic_img = batch['clinic_img'].to(device)
                derm_img = batch['derm_img'].to(device)
                disease_labels = batch['label_disease'].to(device)
                concept_labels = batch['concept_labels'].to(device)
                
                # Đẩy Metadata lên GPU
                meta_features = batch['metadata'].to(device)
                
                # [SỬA LỖI TRỌNG YẾU]: Bắt buộc phải truyền meta_features vào lúc Val
                disease_out, concept_out = model(clinic_img, derm_img, meta_features=meta_features)
                
                loss, _, _ = compute_multitask_loss(
                    disease_out, disease_labels, concept_out, concept_labels, 
                    disease_weights=disease_weights, concept_pos_weights=concept_pos_weights
                )
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nKết thúc Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Đã lưu mô hình tốt nhất tại Epoch {epoch+1}")