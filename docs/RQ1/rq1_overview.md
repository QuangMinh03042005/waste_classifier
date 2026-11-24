# RQ1 – Kiến trúc mô hình: Custom CNN và MobileNetV3_small

> Tài liệu tổng hợp cho báo cáo RQ1. Kết quả final sẽ cập nhật trong `rq1_results.md`.
> Tại thời điểm viết, test in-domain chưa được sử dụng.

## 1. Tóm tắt

RQ1 kiểm chứng liệu transfer learning với MobileNetV3_small có cải thiện hiệu quả
phân loại rác thải so với Custom CNN hay không, và tách ảnh hưởng của **kiến trúc**
khỏi **pretrained weights** bằng ba nhánh:

1. Custom CNN — baseline từ đầu.
2. MobileNetV3_small scratch — kiến trúc MobileNet, không pretrained.
3. MobileNetV3_small pretrained — ImageNet, fine-tune.

Ngoài test in-domain, RQ1 đánh giá thêm trên GD như external test sau khi khóa mô hình.

## 2. Câu hỏi, giả thuyết và tiêu chí

### Câu hỏi

Transfer learning với MobileNetV3_small có cải thiện Macro-F1 so với Custom CNN
trên cùng split không? Phần cải thiện đến từ kiến trúc hay pretrained weights?

### Giả thuyết

- MobileNetV3_small pretrained cao hơn Custom CNN ít nhất 0.02.
- Kết quả ổn định trên ít nhất 4/5 seed.
- Pretrained cao hơn scratch; scratch có thể cao hơn Custom CNN.
- Custom CNN có FLOPs cao hơn đáng kể so với MobileNetV3_small.

### Tiêu chí kết luận

- Metric chính: Macro-F1 test.
- Mean ± std trên 5 seed; so sánh từng seed.
- Báo cáo accuracy, balanced accuracy, per-class F1, confusion matrix.
- Báo cáo params, GFLOPs, latency, VRAM.
- Ngưỡng 0.02 tương đương khoảng 16 ảnh đúng trên test 823 ảnh; đủ lớn để không
  bị nhiễu seed.
- Nếu chênh lệch nhỏ hơn biến động giữa seed, kết luận là “không đủ bằng chứng
  khác biệt”, không khẳng định mô hình này hơn mô hình kia.

## 3. Dữ liệu

### Nguồn

| Dataset | Vai trò | Ảnh dùng 5 lớp | License |
|---|---|---:|---|
| RealWaste | Train/val/test | 3.092 | CC BY 4.0 |
| TrashNet | Train/val/test | 2.390 | MIT |
| GD | External test RQ1 | 4.806 sau dedup | MIT |

### Phân bố 5 lớp

| Lớp | RealWaste | TrashNet | Tổng |
|---|---:|---:|---:|
| cardboard | 461 | 403 | 864 |
| glass | 420 | 501 | 921 |
| metal | 790 | 410 | 1.200 |
| paper | 500 | 594 | 1.094 |
| plastic | 921 | 482 | 1.403 |
| **Tổng** | **3.092** | **2.390** | **5.482** |

- Split v1: 3.833 train / 826 validation / 823 test.
- 70/15/15, seed 42, stratified theo `(dataset, class)`.
- pHash gom ảnh trùng; 35 ảnh trùng, không tách group giữa các split.
- Imbalance ratio: 1,62; dùng Macro-F1 thay vì accuracy.

### EDA

- RealWaste 524×524, nền thực tế; TrashNet 512×384, nền đơn giản.
- 20 ảnh mờ, 4 ảnh quá sáng; không có ảnh hỏng/quá nhỏ.
- 7 group ảnh trùng có nhãn xung đột (15 ảnh) — nhiễu nhãn nhỏ.
- Chi tiết: `docs/EDA/eda_report.md`.

### GD external test

- 7.010 ảnh GD mapping 5 lớp.
- Loại 2.200 ảnh trùng với split v1 và 4 ảnh trùng nội bộ.
- Còn 4.806 ảnh external test.
- Không dùng GD để train hay chọn mô hình.
- Chi tiết: `docs/RQ1/external_validation.md`.

## 4. Kiến trúc

### Custom CNN

| Stage | Thành phần | Output | Params | MACs |
|---|---|---|---:|---:|
| `stem` | Conv 3×3 (3→32), BN, ReLU — không pool | 32×224×224 | 928 | 43,4 M |
| `stages.0` | Conv 3×3 (32→32), BN, ReLU, MaxPool | 32×112×112 | 9.280 | 462,4 M |
| `stages.1` | Conv 3×3 (32→64) → Conv 3×3 (64→64), BN, ReLU, MaxPool | 64×56×56 | 55.552 | 693,6 M |
| `stages.2` | Conv 3×3 (64→128) → Conv 3×3 (128→128), BN, ReLU, MaxPool | 128×28×28 | 221.696 | 693,6 M |
| `stages.3` | Conv 3×3 (128→256) → Conv 3×3 (256→256), BN, ReLU, MaxPool | 256×14×14 | 885.760 | 693,6 M |
| Pool | AdaptiveAvgPool2d(1) | 256 | 0 | – |
| Head | Dropout 0.3 → Linear(256→5) | 5 logits | 1.285 | <0,01 M |

- Params: 1.174.501; FLOPs đo được: 5,17 GFLOPs/ảnh 224×224 (2,59 GMAC).
- Conv 3×3 dùng `padding=1` nên giữ nguyên H×W; chỉ MaxPool2d(2) giảm chiều.
- `stem` không pool; lần giảm chiều đầu tiên nằm ở `stages.0`.
- Tên stage khớp module trong code (`model.stem`, `model.stages[i]`).
- Bảng layer-by-layer: `docs/RQ1/custom_cnn_layers.md` (`scripts/model_summary.py`).
- Residual variant: 1.218.405 params, 5,33 GFLOPs; không được chọn.
- Cấu hình cuối: 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3, 50 epoch.

### MobileNetV3_small

- Dùng implementation có sẵn trong `torchvision` (`mobilenet_v3_small`), không tự viết lại.
- Thành phần: depthwise separable conv, inverted residual, Squeeze-and-Excitation,
  hard-swish/hard-sigmoid.
- 11 block inverted residual: ảnh 224×224×3 → `7×7×576`; GAP; classifier.
- Classifier gốc `576→1024→1000`, thay bằng `Dropout(0.2)` + `Linear(1024→5)`.
- Pretrained `IMAGENET1K_V1`; scratch dùng `weights=None` trên cùng định nghĩa.
- Scratch final dùng patience 20 sau khi chẩn đoán early stopping; 5 seed đạt
  0,6734 ± 0,0144 test Macro-F1, không còn seed collapse.
- Params: 1.522.981; FLOPs: 0,111 GFLOPs/ảnh (2,59 GMAC của Custom CNN).
- Pretrained: full fine-tune (`freeze_epochs = 0`), dropout 0.2.
- Cấu hình pretrained được chọn trên validation; M1 full fine-tune thắng M4.

### So sánh quy mô

| Mô hình | Params | GFLOPs |
|---|---:|---:|
| Custom CNN | 1.174.501 | 5,17 |
| MobileNetV3_small | 1.522.981 | 0,11 |

Số tham số tương đương nhưng Custom CNN nặng khoảng 46 lần FLOPs. Vì vậy RQ1 phải
báo cáo cả chi phí tính toán, không chỉ accuracy.

## 5. Thiết kế thực nghiệm

### Ba arm

| Arm | Mô tả | Pretrained |
|---|---|---|
| C | Custom CNN từ đầu | Không |
| S | MobileNetV3_small scratch | Không |
| P | MobileNetV3_small ImageNet | Có |

### Biến cố định

- Split v1, input 224×224, normalize ImageNet.
- Không augmentation, không class weighting.
- AdamW, batch 64, weight decay 1e-4, cosine scheduler.
- 50 epoch, early stopping theo validation Macro-F1.
- 5 seed cố định: 42, 123, 2024, 7, 3407. Các giá trị được chọn trước khi chạy, dùng giống nhau cho cả 3 arm và không mang ý nghĩa thống kê.
- Cùng code, metric và quy trình đánh giá.

### Quy trình

1. Chọn cấu hình Custom CNN trên validation: 4 ứng viên + 6 debug + residual.
2. Chọn cấu hình MobileNet pretrained trên validation: full fine-tune, freeze 3,
   freeze 5, dropout 0.3; xác nhận top-2 ở 50 epoch.
3. Khóa cấu hình.
4. Chạy 3 arm × 5 seed.
5. Đánh giá test một lần.
6. Đánh giá external GD.
7. Phân tích lỗi, Grad-CAM, efficiency.

### Chống leakage và HARKing

- Mọi lựa chọn dùng validation.
- Test chỉ đánh giá một lần sau khi khóa.
- Plan được commit trước khi chạy.
- Không đổi tiêu chí sau khi xem test.
- External GD không dùng để chọn model.

## 6. Độ đo

### Phân loại

- Accuracy.
- Balanced accuracy.
- Precision/recall/F1 macro.
- Per-class F1, recall, support.
- Confusion matrix.

### Hiệu năng

- Params.
- GFLOPs cho một ảnh 224×224.
- Latency batch 1 và batch 32.
- Throughput ảnh/giây.
- VRAM tối đa khi suy luận.
- Thời gian huấn luyện.

### Thống kê

- Mean ± std trên 5 seed.
- Paired difference theo seed.
- Bootstrap CI cho chênh lệch Macro-F1 nếu thực hiện.
- Không dùng p-value đơn lẻ để khẳng định khi chỉ có 5 seed.

### External validation

- So sánh in-domain Macro-F1 với GD Macro-F1.
- Báo cáo “drop” = in-domain − external.
- Per-class F1 và confusion matrix trên GD.
- Grad-CAM trên ảnh GD sai.

## 7. Cách đánh giá

1. Chọn epoch theo validation Macro-F1; không dùng test.
2. So sánh 3 arm theo mean ± std và từng seed.
3. Phân rã:
   - S > C: kiến trúc đóng góp.
   - P > S: pretrained đóng góp.
   - P ≈ S: pretrained không tạo khác biệt.
4. Phân tích confusion matrix, per-class F1, Grad-CAM.
5. So sánh accuracy/Macro-F1 trên mỗi GFLOPs và latency.
6. External GD để kiểm tra robustness ngoài miền.
7. Kết luận trung thực kể cả khi giả thuyết sai.

## 8. Trạng thái và kế hoạch

### Đã hoàn thành

- Data pipeline, split v1, EDA.
- Custom CNN và MobileNetV3_small.
- Training engine, logging, checkpoint, resume.
- Chọn và debug Custom CNN trên validation.
- Chọn MobileNet pretrained trên validation: M1 full fine-tune.
- Chạy final Custom CNN và MobileNet pretrained, 5 seed.
- Chẩn đoán early stopping: patience 7 cắt sớm 2 seed scratch.
- Chạy final scratch với patience 20, 5 seed (mean 0,6734 ± 0,0144).
- Phân tích test, per-class, confusion matrix, efficiency, Grad-CAM.
- External validation GD cho Custom và pretrained.
- Công cụ phân tích, efficiency, Grad-CAM, feature map.

### Kết quả test tóm tắt

| Arm | Test Macro-F1 | GD Macro-F1 | Drop |
|---|---:|---:|---:|
| Custom CNN | 0,8870 ± 0,0060 | 0,3694 ± 0,0117 | 0,5175 |
| MobileNetV3_small pretrained | 0,8901 ± 0,0114 | 0,5409 ± 0,0097 | 0,3492 |
| MobileNetV3_small scratch (p20) | 0,6734 ± 0,0144 | 0,2834 ± 0,0230 | 0,3900 |

### Chưa hoàn thành

- Hoàn thiện slide RQ1.

### Kế hoạch chạy

1. Làm slide RQ1 từ kết quả trong `rq1_results.md` và báo cáo.
2. Rà soát báo cáo trước khi nộp.

## 9. Hạn chế

- Dataset 5.482 ảnh, tương đối nhỏ cho deep learning.
- Mất cân bằng lớp 1,62.
- Khác biệt domain RealWaste/TrashNet.
- 7 group nhãn xung đột.
- TrashNet 4:3 có thể bị cắt khi CenterCrop 224.
- So sánh kiến trúc khác FLOPs, không chỉ params.
- Chỉ 5 seed, test một lần cho mỗi phiên bản.
- Scratch phải sửa recipe hai lần (learning rate, early stopping); test bị tái sử dụng
  và đã công bố.
- Patience của scratch là 20, của Custom/pretrained là 7; ngân sách tối đa 50 epoch
  cho cả ba arm.
- External GD có 31,4% ảnh trùng train, đã loại; phần còn lại vẫn có domain shift.
- RQ1 không dùng augmentation/class weighting do thiết kế.

## 10. Tài liệu liên quan

| File | Nội dung |
|---|---|
| `docs/RQ1/rq1_debug.md` | Debug Custom CNN |
| `docs/RQ1/customcnn_debug.md` | Báo cáo debug sinh tự động |
| `docs/RQ1/model_selection.md` | Chọn cấu hình Custom CNN |
| `docs/RQ1/mobilenet_selection.md` | Chọn cấu hình MobileNet |
| `docs/RQ1/external_validation.md` | External validation GD |
| `docs/RQ1/rq1_results.md` | Kết quả final |
| `docs/RQ1/rq1_efficiency.md` | Params, FLOPs, latency |
| `docs/notebooks/rq1/feature_maps.pdf` | Feature map qua từng lớp (notebook) |
| `docs/models.md`, `docs/training.md` | Kiến trúc và cách chạy |
| `plans/rq1/model_comparison.yaml` | Plan RQ1 chính |
| `plans/rq1/scratch_diag.yaml` | Chẩn đoán early stopping scratch |
| `plans/rq1/scratch_final_v2.yaml` | Chạy lại scratch với patience 20 |
| `docs/splits.md`, `docs/EDA/eda_report.md` | Dữ liệu và EDA |

## 11. Ghi chú về hỗ trợ AI

Một phần code, tài liệu và plan được hỗ trợ bởi AI. Nhóm đã kiểm tra, chạy thử,
đối chiếu kết quả và chịu trách nhiệm nội dung. Kết quả thực nghiệm chỉ lấy từ run
thật, không lấy từ văn bản do AI sinh.
