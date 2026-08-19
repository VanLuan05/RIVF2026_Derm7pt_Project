# CONTRIBUTION 1 - FINAL SUMMARY

## 1. Mục tiêu

Contribution 1 xây dựng kiến trúc đa phương thức sử dụng metadata-guided
Cross-Attention để hợp nhất ảnh clinical, ảnh dermoscopic và metadata lâm sàng
có cấu trúc.

Mục tiêu chính là đánh giá liệu cơ chế Cross-Attention có thể cung cấp một
phương án fusion thay thế cho feature concatenation truyền thống.

---

## 2. Kiến trúc đề xuất

Tên mô hình:

> C1_CrossAttention

Dữ liệu đầu vào:

- Clinical image
- Dermoscopic image
- Structured clinical metadata

Cơ chế Cross-Attention:

- Query: metadata token
- Key: clinical + dermoscopic spatial tokens
- Value: clinical + dermoscopic spatial tokens
- Fusion: Metadata-guided Cross-Attention

Baseline so sánh:

> B5_Dual_Metadata

với cơ chế fusion:

> Feature Concatenation

---

## 3. Training Protocol

- Seeds: 42, 100, 2026
- Maximum epochs: 20
- Learning rate: 5e-5
- Weight decay: 1e-4
- Early stopping patience: 5
- Checkpoint selection: Validation Disease Macro-F1

| Seed | Best Epoch | Best Validation Macro-F1 | Best Val Loss |
|---|---:|---:|---:|
| 42 | 7 | 0.6213 | 1.0129 |
| 100 | 5 | 0.5899 | 0.9135 |
| 2026 | 10 | 0.6053 | 1.1185 |

Model selection is performed on Validation only.

The independent Test set is used only for final evaluation.

---

## 4. Independent Test Results

| Model | Fusion | Macro-F1 | Balanced Accuracy | AUROC |
|---|---|---:|---:|---:|
| B5_Dual_Metadata | Concatenation | 0.5334 ± 0.0213 | 0.5779 ± 0.0376 | 0.8852 ± 0.0103 |
| C1_CrossAttention | Cross-Attention | **0.5410 ± 0.0241** | 0.5699 ± 0.0482 | 0.8263 ± 0.0226 |

Per-seed C1 results:

| Seed | Macro-F1 | Balanced Accuracy | AUROC |
|---|---:|---:|---:|
| 42 | 0.5405 | 0.5812 | 0.8124 |
| 100 | 0.5653 | 0.6113 | 0.8140 |
| 2026 | 0.5171 | 0.5171 | 0.8524 |

C1 achieves a numerically slightly higher mean Macro-F1 than B5.

However:

- B5 has higher Balanced Accuracy;
- B5 has substantially higher AUROC.

Therefore, no claim is made that Cross-Attention is globally superior to
feature concatenation.

No statistical significance claim is made from the three-seed comparison.

---

## 5. Computational Efficiency

Efficiency was measured using:

- GPU: Tesla T4
- Batch size: 1
- Fixed Test sample
- Warm-up iterations: 30
- Timed iterations: 100
- Checkpoint seed: 42
- model.eval()
- torch.inference_mode()
- CUDA synchronization before and after timing

| Model | Parameters | Latency | Throughput | Peak GPU Memory |
|---|---:|---:|---:|---:|
| B5_Dual_Metadata | 49,136,869 | 10.8873 ms/sample | 91.85 samples/s | 423.15 MB |
| C1_CrossAttention | **48,739,045** | 11.4859 ms/sample | 87.06 samples/s | **422.54 MB** |

C1 uses approximately 0.81% fewer parameters than B5 and has nearly identical
peak GPU memory usage.

However, under this controlled benchmark, C1 has slightly higher inference
latency and slightly lower throughput than B5.

Therefore, the current results do **not** support a claim that Cross-Attention
is computationally faster than the concatenation baseline.

The efficiency measurements should be interpreted as hardware- and
implementation-dependent descriptive measurements rather than universal
runtime guarantees.

---

## 6. Kết luận C1

C1_CrossAttention successfully implements a genuine metadata-guided
Cross-Attention mechanism in which metadata acts as the query and clinical
plus dermoscopic spatial features act as keys and values.

Compared with B5_Dual_Metadata:

- C1 achieves a slightly higher numerical Macro-F1;
- C1 has lower Balanced Accuracy;
- C1 has lower AUROC;
- C1 uses slightly fewer trainable parameters;
- GPU memory usage is nearly identical;
- C1 is slightly slower in the current batch-size-1 inference benchmark.

Therefore, Contribution 1 should be described as:

> an alternative multimodal fusion architecture based on metadata-guided
> Cross-Attention

rather than as a universally superior or computationally faster replacement
for feature concatenation.

**Contribution 1: Metadata-Guided Cross-Attention — COMPLETED.**

---

## 7. Files liên quan

### Source code

- `src/models/models.py`
- `src/run_c1_cross_attention.py`
- `src/evaluate_c1.py`
- `src/c1_efficiency_benchmark.py`

### Checkpoints

- `outputs/C1_CrossAttention_seed_42.pth`
- `outputs/C1_CrossAttention_seed_100.pth`
- `outputs/C1_CrossAttention_seed_2026.pth`

### Results

- `results/contribution_1/c1_seed_results.csv`
- `results/contribution_1/c1_comparison.csv`
- `results/contribution_1/c1_efficiency.csv`
- `results/contribution_1/c1_final_results.md`
- `results/contribution_1/c1_training_manifest_seed_42.json`
- `results/contribution_1/c1_training_manifest_seed_100.json`
- `results/contribution_1/c1_training_manifest_seed_2026.json`
