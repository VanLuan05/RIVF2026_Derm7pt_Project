import torch
import torch.nn as nn
from torchvision import models

class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes=5, num_concepts=7, modality='dual', bottleneck_type='hybrid', use_metadata=False):
        """
        modality: 'clinic_only' (B1), 'derm_only' (B0), 'dual' (B2-B5)
        bottleneck_type: 'none' (B0-B2), 'pure' (B4 - Pure CBM), 'hybrid' (B3, B5 - Hybrid CBM)
        """
        super(MultimodalDermModel, self).__init__()
        
        self.modality = modality
        self.bottleneck_type = bottleneck_type
        self.use_metadata = use_metadata
        
        # 1. KHỞI TẠO BACKBONE (ResNet50) TÙY THEO MODALITY
        if self.modality in ['clinic_only', 'dual']:
            resnet_clinic = models.resnet50(pretrained=True)
            self.clinic_backbone = nn.Sequential(*list(resnet_clinic.children())[:-1]) # Xóa lớp FC cuối
            
        if self.modality in ['derm_only', 'dual']:
            resnet_derm = models.resnet50(pretrained=True)
            self.derm_backbone = nn.Sequential(*list(resnet_derm.children())[:-1])

        # 2. TÍNH TOÁN KÍCH THƯỚC FEATURE
        self.feature_dim = 2048 if self.modality in ['clinic_only', 'derm_only'] else 4096
        
        # (Lỗi 4) Chuẩn bị chỗ trống cho Metadata Encoder (sẽ làm ở P2)
        if self.use_metadata:
            self.meta_dim = 32 # Giả sử Metadata được nén về vector 32 chiều
            self.feature_dim += self.meta_dim
            # self.meta_encoder = ... (Sẽ triển khai sau)

        # 3. KHỞI TẠO BỘ DỰ ĐOÁN KHÁI NIỆM (CHỈ CÓ Ở B3, B4, B5)
        if self.bottleneck_type in ['pure', 'hybrid']:
            self.concept_classifier = nn.Linear(self.feature_dim, num_concepts)

        # 4. KHỞI TẠO BỘ DỰ ĐOÁN BỆNH (GIẢI QUYẾT LỖI 5)
        if self.bottleneck_type == 'none':
            # B0, B1, B2: Dự đoán bệnh thẳng từ đặc trưng ảnh
            disease_in_features = self.feature_dim
            
        elif self.bottleneck_type == 'pure':
            # B4 (Pure CBM): Dự đoán bệnh CHỈ TỪ 7 xác suất khái niệm
            disease_in_features = num_concepts
            
        elif self.bottleneck_type == 'hybrid':
            # B3, B5 (Hybrid CBM): Nối đặc trưng ảnh và 7 xác suất khái niệm
            disease_in_features = self.feature_dim + num_concepts
        else:
            raise ValueError("bottleneck_type phải là 'none', 'pure', hoặc 'hybrid'")

        # Phân loại bệnh (Classifier)
        self.disease_classifier = nn.Sequential(
            nn.Linear(disease_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clinic_img, derm_img, meta_features=None):
        features = []
        
        # 1. Trích xuất đặc trưng ảnh
        if self.modality in ['clinic_only', 'dual']:
            c_feat = self.clinic_backbone(clinic_img).view(clinic_img.size(0), -1)
            features.append(c_feat)
            
        if self.modality in ['derm_only', 'dual']:
            d_feat = self.derm_backbone(derm_img).view(derm_img.size(0), -1)
            features.append(d_feat)
            
        # Ghép đặc trưng (nếu là dual thì sẽ ghép c_feat và d_feat)
        combined_features = torch.cat(features, dim=1)

        # 2. Xử lý Metadata (Sẽ tích hợp sau)
        # if self.use_metadata and meta_features is not None:
        #     meta_out = self.meta_encoder(meta_features)
        #     combined_features = torch.cat((combined_features, meta_out), dim=1)

        # 3. Phân luồng Cấu trúc (Bottleneck Routing)
        if self.bottleneck_type == 'none':
            # Baseline B0, B1, B2 (Không có Concepts)
            disease_logits = self.disease_classifier(combined_features)
            return disease_logits, None
            
        elif self.bottleneck_type == 'pure':
            # Baseline B4 (Pure CBM)
            concept_logits = self.concept_classifier(combined_features)
            concept_probs = torch.sigmoid(concept_logits)
            disease_logits = self.disease_classifier(concept_probs) # CHỈ dùng concept
            return disease_logits, concept_logits
            
        elif self.bottleneck_type == 'hybrid':
            # Baseline B3, B5 (Hybrid CBM)
            concept_logits = self.concept_classifier(combined_features)
            concept_probs = torch.sigmoid(concept_logits)
            hybrid_features = torch.cat((combined_features, concept_probs), dim=1) # Dùng cả hai
            disease_logits = self.disease_classifier(hybrid_features)
            return disease_logits, concept_logits