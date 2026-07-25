import torch
import torch.nn as nn
import torchvision.models as models

# 1. Module mã hóa Metadata (MLP)
class MetaEncoder(nn.Module):
    def __init__(self, num_categorical_features, hidden_dim=64):
        super(MetaEncoder, self).__init__()
        # Ví dụ đơn giản: Giả sử metadata đã được chuyển thành số (one-hot hoặc số nguyên)
        # Trong thực tế, bạn sẽ cần lớp Embedding cho dữ liệu Categorical
        self.fc = nn.Sequential(
            nn.Linear(num_categorical_features, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
    def forward(self, x):
        return self.fc(x)

# 2. Kiến trúc chính: Multimodal Fusion Model
class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes=20, num_concepts=7, use_metadata=True):
        super(MultimodalDermModel, self).__init__()
        self.use_metadata = use_metadata
        
        # Nhánh 1: Trích xuất đặc trưng từ ảnh Clinic (Dùng ResNet50 bỏ lớp cuối)
        resnet1 = models.resnet50(pretrained=True)
        self.clinic_encoder = nn.Sequential(*list(resnet1.children())[:-1]) # Output shape: (Batch, 2048, 1, 1)
        
        # Nhánh 2: Trích xuất đặc trưng từ ảnh Derm (Dùng một ResNet50 khác để học độc lập)
        resnet2 = models.resnet50(pretrained=True)
        self.derm_encoder = nn.Sequential(*list(resnet2.children())[:-1])
        
        feature_dim = 2048 * 2
        
        # Nhánh 3 (Tùy chọn): Metadata Encoder
        if self.use_metadata:
            self.meta_encoder = MetaEncoder(num_categorical_features=3, hidden_dim=64)
            feature_dim += 64
            
        # Lớp phân loại Concept (Nút thắt khái niệm - CBM)
        self.concept_classifier = nn.Linear(feature_dim, num_concepts)
        
        # Lớp phân loại Bệnh (Dự đoán cuối cùng)
        # Tại đây, chúng ta nạp cả đặc trưng đa phương thức + các khái niệm lâm sàng
        self.disease_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim + num_concepts, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, clinic_img, derm_img, meta_features=None):
        # 1. Trích xuất đặc trưng ảnh
        feat_clinic = self.clinic_encoder(clinic_img).squeeze(-1).squeeze(-1) # -> [Batch, 2048]
        feat_derm = self.derm_encoder(derm_img).squeeze(-1).squeeze(-1)       # -> [Batch, 2048]
        
        # 2. Hợp nhất bước 1 (Concatenate cơ bản - Có thể nâng cấp lên Cross-Attention sau)
        if self.use_metadata and meta_features is not None:
            feat_meta = self.meta_encoder(meta_features)
            combined_features = torch.cat((feat_clinic, feat_derm, feat_meta), dim=1)
        else:
            combined_features = torch.cat((feat_clinic, feat_derm), dim=1)
            
        # 3. Dự đoán Khái niệm (Concept Bottleneck)
        concept_logits = self.concept_classifier(combined_features)
        
        # 4. Dự đoán Bệnh (Bằng cách ghép nối đặc trưng chung + xác suất của các Concept)
        # Sigmoid để đưa concept về xác suất [0,1]
        concept_probs = torch.sigmoid(concept_logits) 
        final_input = torch.cat((combined_features, concept_probs), dim=1)
        
        disease_logits = self.disease_classifier(final_input)
        
        return disease_logits, concept_logits

# Khối Test nhanh xem mô hình có bị lỗi kích thước không
if __name__ == "__main__":
    # Tạo dữ liệu giả lập (Dummy data) giống như dữ liệu trả ra từ DataLoader của bạn
    dummy_clinic = torch.randn(16, 3, 224, 224)
    dummy_derm = torch.randn(16, 3, 224, 224)
    dummy_meta = torch.randn(16, 3) # Giả định metadata có 3 cột
    
    # Khởi tạo mô hình (Giả sử dự đoán 20 bệnh, 7 khái niệm)
    model = MultimodalDermModel(num_classes=20, num_concepts=7)
    
    print("Bắt đầu cho dữ liệu chạy qua mô hình...")
    disease_out, concept_out = model(dummy_clinic, dummy_derm, dummy_meta)
    
    print(f"Kích thước dự đoán Bệnh: {disease_out.shape} -> (Nên là [16, 20])")
    print(f"Kích thước dự đoán Khái niệm: {concept_out.shape} -> (Nên là [16, 7])")