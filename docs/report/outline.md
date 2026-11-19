# Cấu trúc báo cáo cuối kỳ

**Đề tài:** Waste Classification Using CNN  
**Mục tiêu độ dài:** khoảng 40 trang (không tính bìa, mục lục, tài liệu tham khảo, phụ lục)  
**Ngôn ngữ:** tiếng Việt, giữ nguyên thuật ngữ tiếng Anh khi cần.

> Quy ước: các mục đánh dấu `[TODO]` sẽ điền sau khi có kết quả final.
> Không đưa số liệu validation vào phần kết quả test.

---

## Trang đầu

- Bìa: trường, môn học, đề tài, nhóm, thành viên, giảng viên, ngày.
- Tóm tắt báo cáo: 200–300 từ. `[TODO: điền sau khi có kết quả RQ1–RQ3]`
- Mục lục.
- Danh mục viết tắt.
- Danh mục bảng.
- Danh mục hình.

---

## Chương 1. Giới thiệu (~5 trang)

### 1.1. Bối cảnh và động lực
- Vai trò phân loại rác thải trong quản lý và tái chế.
- Hạn chế của phân loại thủ công.
- Vai trò của CNN trong phân loại ảnh.
- Lý do chọn bài toán 5 lớp.

### 1.2. Vấn đề nghiên cứu
- Bài toán phân loại đa lớp.
- Thách thức: imbalance, domain shift, chi phí tính toán.
- Khoảng trống: so sánh công bằng giữa CNN tự xây dựng và transfer learning; tách vai trò kiến trúc và pretrained.

### 1.3. Mục tiêu
- Xây dựng baseline Custom CNN.
- So sánh 3 arm: Custom CNN, MobileNetV3_small scratch, MobileNetV3_small pretrained.
- Đánh giá Data Augmentation và Class Weighting.
- Đánh giá cross-dataset và external GD.
- Phân tích lỗi, Grad-CAM, efficiency.
- Xây dựng demo Streamlit.

### 1.4. Câu hỏi nghiên cứu và giả thuyết

Bảng:

| RQ | Câu hỏi | Giả thuyết | Đối chứng | Metric | Tiêu chí kết luận |
|---|---|---|---|---|---|
| RQ1 | ... | ... | ... | Macro-F1 | ... |
| RQ2 | ... | ... | ... | Macro-F1, per-class recall | ... |
| RQ3 | ... | ... | ... | Macro-F1 | ... |

### 1.5. Input, output và phạm vi
- Input: ảnh RGB 224×224.
- Output: 5 lớp, xác suất.
- Phạm vi: 5 lớp, RealWaste + TrashNet; GD external.
- Không làm: detection, segmentation, multi-label.

### 1.6. Đóng góp
- Baseline có kiểm soát.
- So sánh 3 arm, phân rã kiến trúc/pretrained.
- EDA, split chống leakage.
- External validation GD.
- Phân tích efficiency, lỗi, Grad-CAM.
- Demo.

### 1.7. Cấu trúc báo cáo
- Tóm tắt 5 chương.

---

## Chương 2. Cơ sở lý thuyết và nghiên cứu liên quan (~8 trang)

### 2.1. CNN cơ bản
- Convolution, padding, stride, receptive field.
- Pooling.
- BatchNorm.
- ReLU/Hardswish.
- Dropout, weight decay.
- Global Average Pooling.

### 2.2. Transfer learning và fine-tuning
- Pretrained ImageNet.
- Feature extraction vs fine-tuning.
- Freeze/unfreeze.
- Learning rate backbone/head.
- Khi nào transfer learning có lợi.

### 2.3. MobileNetV3_small
- Depthwise separable convolution.
- Inverted residual.
- Squeeze-and-Excitation.
- Hardswish/Hardsigmoid.
- Chi phí tính toán.

### 2.4. Custom CNN
- Thiết kế stage.
- Kaiming/Xavier initialization.
- So sánh params/FLOPs với MobileNet.

### 2.5. Tối ưu và huấn luyện
- CrossEntropyLoss.
- AdamW.
- Cosine annealing.
- AMP.
- Early stopping.
- Checkpoint/resume.

### 2.6. Độ đo
- Accuracy, balanced accuracy.
- Precision/recall/F1 macro.
- Per-class F1.
- Confusion matrix.
- Params, FLOPs, latency, VRAM.

### 2.7. Giải thích mô hình
- Grad-CAM: nguyên lý.
- Layer mục tiêu cho Custom CNN và MobileNet.
- Ứng dụng phân tích lỗi.

### 2.8. Nghiên cứu liên quan
- RealWaste paper.
- TrashNet.
- MobileNetV3.
- Các nghiên cứu waste classification.
- Khoảng trống và lý do chọn giải pháp.

### 2.9. Kết luận chương
- Cơ sở cho 3 arm và protocol.

---

## Chương 3. Dữ liệu và phương pháp (~12 trang)

### 3.1. Dataset
- RealWaste: 4.752 ảnh, 9 lớp, CC BY 4.0.
- TrashNet: 2.527 ảnh, 6 lớp, MIT.
- GD: external test, 13.348 ảnh, MIT.
- Bảng tổng hợp nguồn, license, vai trò.

### 3.2. Mapping nhãn
- 5 lớp chung.
- Bảng mapping RealWaste/TrashNet/GD.
- Lớp bị loại và lý do.
- Số ảnh bị loại.

### 3.3. EDA
- Phân bố lớp, imbalance ratio 1,62.
- Phân bố split.
- Ảnh mẫu hai nguồn.
- Kích thước, aspect ratio, brightness, blur.
- Ảnh trùng, 7 group xung đột nhãn.
- Domain shift.
- Nhận xét dẫn tới quyết định preprocessing/metric.

### 3.4. Tiền xử lý
- Resize 256, CenterCrop 224.
- Normalize ImageNet.
- RQ1: không augmentation.
- RQ2: RandomResizedCrop, flip, rotation, jitter, erasing.
- Giải thích tác động.

### 3.5. Chia dữ liệu
- Split v1: 70/15/15, seed 42.
- Stratified theo `(dataset, class)`.
- pHash group, chống leakage.
- Test chỉ dùng một lần.
- GD external: loại 2.200 ảnh trùng train, còn 4.806.

### 3.6. Kiến trúc Custom CNN
- Sơ đồ stage.
- Tensor shape.
- Params, FLOPs.
- Cấu hình được chọn qua validation.
- Residual variant và lý do không chọn.

### 3.7. Kiến trúc MobileNetV3_small
- Backbone, classifier.
- Scratch vs pretrained.
- Freeze/unfreeze.
- M1 full fine-tune được chọn.
- Params, FLOPs.

### 3.8. Pipeline huấn luyện
- DataLoader.
- Loss.
- Optimizer/scheduler.
- AMP.
- Early stopping.
- Checkpoint last/best.
- Logging.

### 3.9. Thiết kế thực nghiệm
- Ma trận RQ–thí nghiệm.
- Biến thay đổi, biến cố định.
- 5 seed RQ1; 3 seed RQ2/RQ3.
- Test một lần.
- Plan pre-registration.

### 3.10. RQ1
- 3 arm.
- 5 seed.
- 50 epoch.
- Không augmentation/class weighting.
- External GD.

### 3.11. RQ2
- Full.
- No augmentation.
- No class weighting.
- 3 seed.

### 3.12. RQ3
- RealWaste → TrashNet.
- TrashNet → RealWaste.
- Pooled → test từng nguồn.
- 3 seed.

### 3.13. External validation
- GD mapping/dedup.
- Chỉ đánh giá sau khi khóa.
- Không dùng chọn model.

### 3.14. Công cụ và tái lập
- Python 3.12, PyTorch, torchvision.
- uv, pytest, ruff.
- Git commit, environment.json.
- Cấu trúc repo.

---

## Chương 4. Thực nghiệm, kết quả và thảo luận (~10 trang)

### 4.1. Môi trường thực nghiệm
- GPU/CPU, VRAM, CUDA, PyTorch.
- Thời gian chạy.
- Phiên bản thư viện. `[TODO]`

### 4.2. RQ1 – Model Architecture
#### 4.2.1. Kết quả chọn cấu hình trên validation
- Bảng chọn Custom CNN.
- Bảng debug.
- Bảng chọn MobileNet.
- Ghi rõ đây không phải kết quả test. `[TODO]`

#### 4.2.2. Kết quả test
- Bảng 3 arm × 5 seed.
- Mean ± std.
- Paired difference.
- Accuracy, balanced accuracy, Macro-F1. `[TODO]`

#### 4.2.3. Per-class và confusion matrix
- Bảng per-class F1.
- Confusion matrix từng arm.
- Nhận xét lớp yếu. `[TODO]`

#### 4.2.4. Learning curves
- Train/val loss.
- Macro-F1 theo epoch.
- Overfitting/underfitting. `[TODO]`

#### 4.2.5. Efficiency
- Params, GFLOPs, latency, VRAM.
- Macro-F1 trên mỗi GFLOPs. `[TODO]`

#### 4.2.6. Phân rã kiến trúc và pretrained
- C vs S vs P.
- P−C, S−C, P−S.
- Kết luận. `[TODO]`

#### 4.2.7. Trả lời RQ1
- Giả thuyết đúng/sai.
- Mức chênh lệch, độ ổn định.
- Đánh đổi compute. `[TODO]`

### 4.3. RQ2 – Data Augmentation & Class Weighting
- Bảng full/noaug/nocw.
- Mean ± std.
- Per-class recall.
- Learning curves.
- Tác động lên lớp thiểu số.
- Trả lời RQ2. `[TODO]`

### 4.4. RQ3 – Generalization
- Bảng cross-dataset.
- Pooled vs single-source.
- Domain shift.
- Per-class/confusion.
- Trả lời RQ3. `[TODO]`

### 4.5. External validation GD
- Bảng in-domain vs GD.
- Drop.
- Per-class.
- Nhận xét robustness. `[TODO]`

### 4.6. Phân tích lỗi và Grad-CAM
- Top confusion pairs.
- Ảnh sai tiêu biểu.
- Grad-CAM Custom CNN vs MobileNet.
- Background bias. `[TODO]`

### 4.7. Thảo luận chung
- Overfitting/underfitting.
- Generalization.
- Ảnh hưởng imbalance.
- Chi phí tính toán.
- So sánh với kỳ vọng.
- Yếu tố gây nhiễu.

### 4.8. Demo ứng dụng
- Kiến trúc Streamlit.
- Input/output.
- Ảnh minh họa.
- Hướng dẫn chạy.
- Giới hạn demo. `[TODO]`

### 4.9. Đe dọa tính hợp lệ
- Dataset nhỏ.
- Domain shift.
- Label noise.
- Test một lần.
- Số seed.
- Khác biệt GPU.

---

## Chương 5. Kết luận và hướng phát triển (~4 trang)

### 5.1. Kết luận theo RQ
- RQ1: kết quả, vai trò kiến trúc/pretrained. `[TODO]`
- RQ2: augmentation/class weighting. `[TODO]`
- RQ3: cross-dataset/pooled. `[TODO]`
- External GD. `[TODO]`

### 5.2. Đóng góp
- Kỹ thuật, thực nghiệm, phân tích, demo.

### 5.3. Hạn chế
- Dữ liệu.
- Thiết kế.
- Tài nguyên.
- Phạm vi.

### 5.4. Hướng phát triển
- Thêm dữ liệu Việt Nam.
- Detection/segmentation.
- Calibration/OOD.
- Triển khai edge.
- Multi-label.

---

## Tài liệu tham khảo

- RealWaste.
- TrashNet.
- MobileNetV3.
- ImageNet.
- AdamW.
- Grad-CAM.
- PyTorch/torchvision/scikit-learn.
- Các bài báo waste classification.

---

## Phụ lục

### Phụ lục A. Dataset và license
- Nguồn, license, citation, số ảnh.

### Phụ lục B. Mapping nhãn
- Bảng mapping đầy đủ, lớp bị loại.

### Phụ lục C. Ma trận thí nghiệm
- Core: 9 experiment.
- Pilot/selection: 17 experiment.

### Phụ lục D. Config và hyperparameter
- Config RQ1/RQ2/RQ3.
- Giải thích lựa chọn.

### Phụ lục E. Log và artifact
- Cấu trúc run dir.
- Ví dụ metrics.json, history.jsonl.
- Checkpoint.

### Phụ lục F. Kết quả bổ sung
- Per-class đầy đủ.
- Confusion matrix.
- Learning curves.
- External GD.

### Phụ lục G. Hướng dẫn tái lập
- Cài môi trường.
- Tải dữ liệu.
- Split/EDA.
- Train/eval.
- Export.

### Phụ lục H. Phân công và đóng góp
- Bảng thành viên, RQ, tỷ lệ đóng góp.
- Minh chứng commit/PR.

### Phụ lục I. Ghi nhận AI
- Phần nào được hỗ trợ.
- Nhóm đã kiểm tra/chạy thử.

---

## Phân bổ trang gợi ý

| Phần | Trang |
|---|---:|
| Tóm tắt + mục lục + danh mục | 3 |
| Chương 1 | 5 |
| Chương 2 | 8 |
| Chương 3 | 12 |
| Chương 4 | 10 |
| Chương 5 | 4 |
| **Tổng phần chính** | **42** |
| Tài liệu tham khảo + phụ lục | 8–10 |

Nếu cần đúng 40 trang, giảm Chương 2 còn 6 trang và gộp một số mục nhỏ ở Chương 4.

---

## Việc cần điền sau khi có kết quả

- [ ] RQ1: bảng 3 arm × 5 seed.
- [ ] RQ1: learning curves, confusion matrix, per-class F1.
- [ ] RQ1: params/GFLOPs/latency.
- [ ] RQ1: external GD.
- [ ] RQ2: bảng full/noaug/nocw.
- [ ] RQ3: bảng cross-dataset/pooled.
- [ ] Grad-CAM.
- [ ] Demo screenshot.
- [ ] Kết luận từng RQ.
- [ ] Tóm tắt báo cáo.

---

## Nguồn tài liệu nội bộ

| Nội dung | File |
|---|---|
| Tổng quan RQ1 | `docs/RQ1/rq1_overview.md` |
| Kết quả RQ1 | `docs/RQ1/rq1_results.md` |
| Chọn Custom CNN | `docs/RQ1/model_selection.md` |
| Chọn MobileNet | `docs/RQ1/mobilenet_selection.md` |
| External GD | `docs/RQ1/external_validation.md` |
| EDA | `docs/EDA/eda_report.md` |
| Dữ liệu/split | `docs/datasets.md`, `docs/splits.md`, `docs/label_mapping.md` |
| Training/log | `docs/training.md`, `docs/logging.md` |
| Ma trận thí nghiệm | `docs/experiment_matrix.md` |
| Plan RQ | `plans/rq1/`, `plans/rq2/`, `plans/rq3/` |
