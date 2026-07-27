import torch
import torch.nn as nn
from torchvision import models

class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes=5, num_concepts=7, modality='dual', bottleneck_type='hybrid', use_metadata=False):
        super(MultimodalDermModel, self).__init__()
        
        self.modality = modality
        self.bottleneck_type = bottleneck_type
        self.use_metadata = use_metadata
        
        # 1. KHỞI TẠO BACKBONE
        if self.modality in ['clinic_only', 'dual']:
            resnet_clinic = models.resnet50(pretrained=True)
            self.clinic_backbone = nn.Sequential(*list(resnet_clinic.children())[:-1]) 
            
        if self.modality in ['derm_only', 'dual']:
            resnet_derm = models.resnet50(pretrained=True)
            self.derm_backbone = nn.Sequential(*list(resnet_derm.children())[:-1])

        # 2. TÍNH TOÁN KÍCH THƯỚC FEATURE
        self.feature_dim = 2048 if self.modality in ['clinic_only', 'derm_only'] else 4096
        
        # ========================================================
        # [GIẢI QUYẾT TỪ MÔ HÌNH]: META-ENCODER ĐÃ KÍCH HOẠT
        # ========================================================
        if self.use_metadata:
            self.meta_dim = 32 
            self.feature_dim += self.meta_dim # Nới rộng chiều kích thước để nhét thêm Meta
            
            # Khởi tạo Mạng Nơ-ron nhỏ để học từ Vector Metadata 32 chiều
            self.meta_encoder = nn.Sequential(
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.meta_dim)
            )

        # 3. KHỞI TẠO BỘ DỰ ĐOÁN KHÁI NIỆM 
        if self.bottleneck_type in ['pure', 'hybrid', 'multitask']: 
            self.concept_classifier = nn.Linear(self.feature_dim, num_concepts)

        # 4. KHỞI TẠO BỘ DỰ ĐOÁN BỆNH 
        if self.bottleneck_type in ['none', 'multitask']: 
            disease_in_features = self.feature_dim
        elif self.bottleneck_type == 'pure':
            disease_in_features = num_concepts
        elif self.bottleneck_type == 'hybrid':
            disease_in_features = self.feature_dim + num_concepts
        else:
            raise ValueError("bottleneck_type không hợp lệ")

        self.disease_classifier = nn.Sequential(
            nn.Linear(disease_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clinic_img, derm_img, meta_features=None):
        features = []
        
        if self.modality in ['clinic_only', 'dual']:
            c_feat = self.clinic_backbone(clinic_img).view(clinic_img.size(0), -1)
            features.append(c_feat)
            
        if self.modality in ['derm_only', 'dual']:
            d_feat = self.derm_backbone(derm_img).view(derm_img.size(0), -1)
            features.append(d_feat)
            
        combined_features = torch.cat(features, dim=1)

        # ========================================================
        # KẾT HỢP DỮ LIỆU NHÂN KHẨU HỌC VÀO ĐẶC TRƯNG ẢNH
        # ========================================================
        if self.use_metadata and meta_features is not None:
            # Cho Metadata đi qua bộ mã hóa
            meta_out = self.meta_encoder(meta_features)
            # Nối (Concatenate) thông tin y tế vào cạnh thông tin hình ảnh
            combined_features = torch.cat((combined_features, meta_out), dim=1)

        # 3. Phân luồng Cấu trúc 
        if self.bottleneck_type == 'none':
            disease_logits = self.disease_classifier(combined_features)
            return disease_logits, None
            
        elif self.bottleneck_type == 'multitask':
            concept_logits = self.concept_classifier(combined_features)
            disease_logits = self.disease_classifier(combined_features) 
            return disease_logits, concept_logits
            
        elif self.bottleneck_type == 'pure':
            concept_logits = self.concept_classifier(combined_features)
            concept_probs = torch.sigmoid(concept_logits)
            disease_logits = self.disease_classifier(concept_probs)
            return disease_logits, concept_logits
            
        elif self.bottleneck_type == 'hybrid':
            concept_logits = self.concept_classifier(combined_features)
            concept_probs = torch.sigmoid(concept_logits)
            hybrid_features = torch.cat((combined_features, concept_probs), dim=1)
            disease_logits = self.disease_classifier(hybrid_features)
            return disease_logits, concept_logits