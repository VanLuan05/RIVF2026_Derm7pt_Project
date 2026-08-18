import torch
import torch.nn as nn
from torchvision import models

class MultimodalDermModel(nn.Module):
    def __init__(self, num_classes=5, num_concepts=7, modality='dual', bottleneck_type='hybrid', use_metadata=False, meta_input_dim=14):
        super(MultimodalDermModel, self).__init__()
        
        self.modality = modality
        self.bottleneck_type = bottleneck_type
        self.use_metadata = use_metadata
        self.meta_input_dim = meta_input_dim

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
                nn.Linear(meta_input_dim, 64),  # <--- SỬA LỖI P0: Nhận đúng 14 chiều từ OneHotEncoder
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


# ================================================================
# CONTRIBUTION 1
# Metadata-Guided Cross-Attention Multimodal Model
# ================================================================

class C1CrossAttentionModel(nn.Module):
    """
    C1 Cross-Attention model.

    Inputs:
        - Clinical image
        - Dermoscopic image
        - Structured metadata

    Cross-Attention:
        Query       = metadata token
        Key / Value = spatial visual tokens

    Output:
        Disease logits
    """

    def __init__(
        self,
        num_classes=5,
        meta_input_dim=14,
        d_model=256,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        # Giữ interface tương thích project hiện tại
        self.modality = "dual"
        self.bottleneck_type = "none"
        self.use_metadata = True
        self.meta_input_dim = meta_input_dim

        # --------------------------------------------------------
        # 1. Clinical image backbone
        # Giữ spatial feature map, KHÔNG dùng AvgPool
        # --------------------------------------------------------

        clinic_resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.clinic_backbone = nn.Sequential(
            *list(clinic_resnet.children())[:-2]
        )

        # --------------------------------------------------------
        # 2. Dermoscopy backbone
        # --------------------------------------------------------

        derm_resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        self.derm_backbone = nn.Sequential(
            *list(derm_resnet.children())[:-2]
        )

        # --------------------------------------------------------
        # 3. Project visual feature: 2048 -> 256
        # --------------------------------------------------------

        self.clinic_projection = nn.Conv2d(
            2048,
            d_model,
            kernel_size=1
        )

        self.derm_projection = nn.Conv2d(
            2048,
            d_model,
            kernel_size=1
        )

        # --------------------------------------------------------
        # 4. Metadata encoder
        #
        # Giữ cấu trúc gần baseline B5:
        # metadata -> 64 -> 32
        # rồi project 32 -> d_model
        # --------------------------------------------------------

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        self.meta_projection = nn.Linear(
            32,
            d_model
        )

        # --------------------------------------------------------
        # 5. CROSS-ATTENTION
        #
        # Q = Metadata
        # K = Visual tokens
        # V = Visual tokens
        # --------------------------------------------------------

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attention_norm = nn.LayerNorm(
            d_model
        )

        # --------------------------------------------------------
        # 6. Lightweight Feed Forward Network
        # --------------------------------------------------------

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.ffn_norm = nn.LayerNorm(
            d_model
        )

        # --------------------------------------------------------
        # 7. Disease classifier
        # --------------------------------------------------------

        self.disease_classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _to_tokens(
        self,
        feature_map,
        projection
    ):
        """
        [B, 2048, H, W]
              ↓
        [B, 256, H, W]
              ↓
        [B, H*W, 256]
        """

        x = projection(feature_map)

        x = x.flatten(2)

        x = x.transpose(1, 2)

        return x

    def forward(
        self,
        clinic_img,
        derm_img,
        meta_features=None,
        intervention_probs=None,
    ):

        if meta_features is None:
            raise ValueError(
                "C1CrossAttentionModel yêu cầu metadata."
            )

        # --------------------------------------------------------
        # Clinical spatial tokens
        # --------------------------------------------------------

        clinic_map = self.clinic_backbone(
            clinic_img
        )

        clinic_tokens = self._to_tokens(
            clinic_map,
            self.clinic_projection
        )

        # --------------------------------------------------------
        # Dermoscopic spatial tokens
        # --------------------------------------------------------

        derm_map = self.derm_backbone(
            derm_img
        )

        derm_tokens = self._to_tokens(
            derm_map,
            self.derm_projection
        )

        # --------------------------------------------------------
        # Visual memory
        #
        # 49 clinical tokens
        # +
        # 49 dermoscopic tokens
        # =
        # 98 visual tokens (input 224x224)
        # --------------------------------------------------------

        visual_tokens = torch.cat(
            [
                clinic_tokens,
                derm_tokens
            ],
            dim=1
        )

        # --------------------------------------------------------
        # Metadata Query token
        # --------------------------------------------------------

        meta_encoded = self.meta_encoder(
            meta_features
        )

        meta_token = self.meta_projection(
            meta_encoded
        ).unsqueeze(1)

        # --------------------------------------------------------
        # REAL CROSS-ATTENTION
        #
        # Q = Metadata
        # K = Clinical + Dermoscopy
        # V = Clinical + Dermoscopy
        # --------------------------------------------------------

        attended_visual, _ = self.cross_attention(
            query=meta_token,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False
        )

        # Metadata residual + attended visual information
        fused_token = self.attention_norm(
            meta_token + attended_visual
        )

        # Feed-forward block
        fused_token = self.ffn_norm(
            fused_token
            + self.ffn(fused_token)
        )

        # [B, 1, 256] -> [B, 256]
        fused_feature = fused_token.squeeze(1)

        # --------------------------------------------------------
        # Diagnosis
        # --------------------------------------------------------

        disease_logits = self.disease_classifier(
            fused_feature
        )

        # Giữ output interface giống project cũ
        return disease_logits, None
