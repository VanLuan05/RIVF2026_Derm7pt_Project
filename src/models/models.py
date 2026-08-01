import torch
import torch.nn as nn
from torchvision import models

class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes=5, num_concepts=7, modality='dual', bottleneck_type='hybrid', use_metadata=False):
        super(MultimodalDermModel, self).__init__()
        
        self.modality = modality
        self.bottleneck_type = bottleneck_type
        self.use_metadata = use_metadata
        
        # 1. KHỞI TẠO BACKBONE XỬ LÝ ẢNH (Bỏ qua hoàn toàn nếu là meta_only)
        if self.modality in ['clinic_only', 'dual']:
            resnet_clinic = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.clinic_backbone = nn.Sequential(*list(resnet_clinic.children())[:-1]) 
            
        if self.modality in ['derm_only', 'dual']:
            resnet_derm = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.derm_backbone = nn.Sequential(*list(resnet_derm.children())[:-1])

        # 2. TÍNH TOÁN KÍCH THƯỚC FEATURE
        if self.modality == 'dual':
            self.feature_dim = 4096
        elif self.modality in ['clinic_only', 'derm_only']:
            self.feature_dim = 2048
        elif self.modality == 'meta_only':
            self.feature_dim = 0
            self.use_metadata = True # Bắt buộc phải bật Metadata nếu không dùng ảnh
        else:
            raise ValueError("Modality không hợp lệ. Chọn: dual, clinic_only, derm_only, meta_only")
        
        # ========================================================
        # KHỞI TẠO META-ENCODER XỬ LÝ NHÂN KHẨU HỌC
        # ========================================================
        if self.use_metadata:
            self.meta_dim = 32 
            self.feature_dim += self.meta_dim 
            
            self.meta_encoder = nn.Sequential(
                nn.Linear(14, 64),  # <--- SỬA LỖI P0: Nhận đúng 14 chiều từ OneHotEncoder
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.meta_dim) # Đầu ra nén thành 32 chiều để nối với ảnh
            )

        # 3. KHỞI TẠO BỘ DỰ ĐOÁN KHÁI NIỆM (Bỏ qua nếu là meta_only)
        if self.bottleneck_type in ['pure', 'hybrid', 'multitask'] and self.modality != 'meta_only': 
            self.concept_classifier = nn.Linear(self.feature_dim, num_concepts)

        # 4. KHỞI TẠO BỘ DỰ ĐOÁN BỆNH 
        if self.modality == 'meta_only':
            disease_in_features = self.feature_dim # Kích thước chính là 32 chiều của MetaEncoder
        else:
            if self.bottleneck_type in ['none', 'multitask']: 
                disease_in_features = self.feature_dim
            elif self.bottleneck_type == 'pure':
                disease_in_features = num_concepts
            elif self.bottleneck_type == 'hybrid':
                disease_in_features = self.feature_dim + num_concepts
            else:
                raise ValueError("Bottleneck_type không hợp lệ")

        self.disease_classifier = nn.Sequential(
            nn.Linear(disease_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clinic_img, derm_img, meta_features=None, intervention_probs=None):
        features = []
        
        # 1. Trích xuất đặc trưng ảnh (Nếu có)
        if self.modality in ['clinic_only', 'dual']:
            c_feat = self.clinic_backbone(clinic_img).view(clinic_img.size(0), -1)
            features.append(c_feat)
            
        if self.modality in ['derm_only', 'dual']:
            d_feat = self.derm_backbone(derm_img).view(derm_img.size(0), -1)
            features.append(d_feat)
            
        if features:
            combined_features = torch.cat(features, dim=1)
        else:
            combined_features = None # Xử lý riêng cho meta_only

        # 2. Xử lý và Hợp nhất Metadata
        if self.use_metadata and meta_features is not None:
            meta_out = self.meta_encoder(meta_features)
            
            if combined_features is not None:
                # Nếu có ảnh thì nối thêm Metadata vào cạnh
                combined_features = torch.cat((combined_features, meta_out), dim=1)
            else:
                # Nếu chạy meta_only thì chỉ dùng duy nhất Metadata
                combined_features = meta_out

        if combined_features is None:
             raise ValueError("Không có dữ liệu nào được trích xuất. Hãy kiểm tra lại đầu vào!")

        # 3. Phân luồng Đầu ra
        # Nếu mô hình chỉ dùng Metadata (Nhắm mắt đoán bệnh)
        if self.modality == 'meta_only':
            disease_logits = self.disease_classifier(combined_features)
            return disease_logits, None 

        # Nếu mô hình có dùng hình ảnh
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
            
            # --- BÁC SĨ CAN THIỆP TẠI ĐÂY ---
            if intervention_probs is not None:
                concept_probs = intervention_probs 
                
            disease_logits = self.disease_classifier(concept_probs)
            return disease_logits, concept_logits
            
        elif self.bottleneck_type == 'hybrid':
            concept_logits = self.concept_classifier(combined_features)
            concept_probs = torch.sigmoid(concept_logits)
            
            # --- BÁC SĨ CAN THIỆP TẠI ĐÂY ---
            if intervention_probs is not None:
                concept_probs = intervention_probs 
                
            # Nối đặc trưng đúng chuẩn
            hybrid_features = torch.cat((combined_features, concept_probs), dim=1)
            disease_logits = self.disease_classifier(hybrid_features)
            
            return disease_logits, concept_logits