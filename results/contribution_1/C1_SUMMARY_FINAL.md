# CONTRIBUTION 1 - FINAL SUMMARY

## 1. Mục tiêu

Xây dựng kiến trúc đa phương thức sử dụng Cross-Attention để hợp nhất
ảnh da và metadata lâm sàng có cấu trúc, đồng thời cải thiện hiệu quả
tính toán so với phương pháp ghép nối đặc trưng truyền thống.

## 2. Kiến trúc đề xuất

Tên mô hình: C1_CrossAttention

Dữ liệu đầu vào:
- Clinical image
- Dermoscopic image
- Structured clinical metadata

Cơ chế fusion:
- Query: metadata token
- Key: clinical + dermoscopic spatial tokens
- Value: clinical + dermoscopic spatial tokens
- Fusion: Metadata-guided Cross-Attention

Baseline so sánh:
- B5_Dual_Metadata
- Fusion: Feature Concatenation

## 3. Training Protocol

- Seeds: 42, 100, 2026
- Epoch tối đa: 20
- Learning rate: 5e-5
- Weight decay: 1e-4
- Early stopping patience: 5
- Checkpoint selection: Validation Disease Macro-F1

| Seed | Best Epoch | Best Validation Macro-F1 | Best Val Loss |
|---|---:|---:|---:|
| 42 | 7 | 0.6213 | 1.0129 |
| 100 | 5 | 0.5899 | 0.9135 |
| 2026 | 10 | 0.6053 | 1.1185 |


## 4. Independent Test Results

| Model | Fusion | Macro-F1 | Balanced Accuracy | AUROC |
|---|---|---:|---:|---:|
| B5_Dual_Metadata | Concatenation | 0.5334 ± 0.0213 | 0.5779 ± 0.0376 | 0.8852 ± 0.0103 |
| C1_CrossAttention | Cross-Attention | 0.5410 ± 0.0241 | 0.5699 ± 0.0482 | 0.8263 ± 0.0226 |

## 5. Computational Efficiency

| Model | Parameters | Latency | Throughput | Peak GPU Memory |
|---|---:|---:|---:|---:|
| B5_Dual_Metadata | 49,136,869 | 15.14 ms | 66.07 samples/s | 1191.74 MB |
| C1_CrossAttention | 48,739,045 | 11.69 ms | 85.52 samples/s | 1192.16 MB |

## 6. Kết luận C1

C1_CrossAttention đã triển khai thành công cơ chế Cross-Attention thực sự
để hợp nhất metadata lâm sàng với đặc trưng không gian từ clinical image
và dermoscopic image.

So với baseline B5 sử dụng concatenation, C1 đạt Macro-F1 trung bình cao
hơn và có latency thấp hơn, throughput cao hơn, trong khi số lượng tham số
gần tương đương.

Tuy nhiên, Balanced Accuracy và AUROC của C1 thấp hơn B5. Vì vậy kết quả
không được diễn giải là Cross-Attention vượt trội trên mọi chỉ số.

Contribution 1 được mô tả là một kiến trúc multimodal Cross-Attention
có hiệu quả tính toán tốt hơn so với feature concatenation baseline.

## 7. Files liên quan

### Source code
- src/models/models.py
- src/run_c1_cross_attention.py
- src/evaluate_c1.py

### Checkpoints
- outputs/C1_CrossAttention_seed_42.pth
- outputs/C1_CrossAttention_seed_100.pth
- outputs/C1_CrossAttention_seed_2026.pth

### Results
- c1_seed_results.csv
- c1_comparison.csv
- c1_efficiency.csv
- c1_final_results.md
- c1_training_manifest_seed_42.json
- c1_training_manifest_seed_100.json
- c1_training_manifest_seed_2026.json
