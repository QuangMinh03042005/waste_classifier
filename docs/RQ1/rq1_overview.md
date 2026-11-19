# RQ1 — Kiến trúc mô hình: Custom CNN và MobileNetV3_small

> Tài liệu tổng hợp phục vụ báo cáo và thuyết trình.
> Kết quả định lượng final được để trống và sẽ cập nhật sau khi chạy xong RQ1.
> Tại thời điểm viết, test set chưa được sử dụng.

## 1. Tóm tắt

RQ1 kiểm chứng liệu transfer learning với MobileNetV3_small có cải thiện hiệu quả
phân loại rác thải so với một CNN tự xây dựng hay không, đồng thời tách ảnh hưởng
của **kiến trúc** khỏi ảnh hưởng của **pretrained weights**.

RQ1 gồm ba nhánh thực nghiệm:

1. **Custom CNN** — huấn luyện từ đầu, đóng vai trò baseline.
2. **MobileNetV3_small scratch** — kiến trúc MobileNet nhưng không dùng pretrained.
3. **MobileNetV3_small pretrained** — transfer learning từ ImageNet.

Cách thiết kế này cho phép trả lời không chỉ “mô hình nào tốt hơn”, mà còn
“vì sao tốt hơn”:

- Nếu pretrained > scratch > custom: lợi ích đến từ cả kiến trúc và pretrained.
- Nếu pretrained ≈ scratch > custom: lợi ích chủ yếu từ kiến trúc.
- Nếu pretrained > custom ≈ scratch: lợi ích chủ yếu từ pretrained weights.

Ngoài độ chính xác, RQ1 còn so sánh chi phí tính toán: số tham số, FLOPs, latency
và bộ nhớ GPU. Đây là phần cần thiết vì Custom CNN và MobileNet có chi phí rất khác
nhau dù số tham số tương đương.

## 2. Câu hỏi nghiên cứu và giả thuyết

### 2.1. Câu hỏi chính

Transfer learning với MobileNetV3_small có cải thiện Macro-F1 so với Custom CNN
huấn luyện từ đầu trên cùng split dữ liệu không?

### 2.2. Câu hỏi phụ

1. Nếu có cải thiện, bao nhiêu đến từ kiến trúc MobileNet và bao nhiêu từ pretrained
   weights?
2. Đánh đổi giữa hiệu quả phân loại và chi phí tính toán là gì?
3. Mô hình nhìn vào đâu để đưa ra dự đoán, và lỗi tập trung ở lớp nào?

### 2.3. Giả thuyết

- MobileNetV3_small pretrained đạt Macro-F1 cao hơn Custom CNN ít nhất 0.02.
- Khoảng cách này ổn định trên ít nhất 4/5 seed.
- MobileNetV3_small pretrained cao hơn MobileNetV3_small scratch.
- Custom CNN có FLOPs cao hơn đáng kể so với MobileNetV3_small.

Giả thuyết có thể bị bác bỏ. Nếu khoảng cách nhỏ hơn 0.02, kết luận sẽ phản ánh
đúng thực tế thay vì cố gắng chứng minh giả thuyết.

### 2.4. Tiêu chí kết luận

- Metric chính: Macro-F1 trên test set.
- Ưu tiên tính ổn định: mean ± std trên 5 seed.
- So sánh theo từng seed (paired difference), không chỉ so sánh trung bình.
- Báo cáo thêm accuracy, balanced accuracy, precision/recall macro, per-class F1,
  confusion matrix và chi phí tính toán.
- Không chọn epoch, threshold hay cấu hình dựa trên test set.

## 3. Phạm vi và đóng góp

### 3.1. Phạm vi

- Bài toán: phân loại ảnh đa lớp.
- 5 lớp chung: cardboard, glass, metal, paper, plastic.
- Dữ liệu: RealWaste và TrashNet.
- Input: ảnh RGB 224×224.
- Output: 5 logits, dự đoán bằng argmax.
- RQ1 không dùng augmentation và không dùng class weighting; hai yếu tố này thuộc RQ2.

### 3.2. Đóng góp

- Xây dựng baseline CNN từ đầu, có kiểm soát capacity và compute.
- So sánh công bằng với MobileNetV3_small pretrained trên cùng split và ngân sách.
- Thêm nhánh scratch để tách vai trò kiến trúc và pretrained.
- Báo cáo đầy đủ accuracy, Macro-F1, per-class F1, confusion matrix, Grad-CAM.
- Báo cáo đầy đủ params, FLOPs, latency, VRAM.
- Thiết kế thực nghiệm có pre-registration, logging và khóa test set.

### 3.3. Không nằm trong phạm vi RQ1

- Data augmentation và class weighting: RQ2.
- Cross-dataset generalization: RQ3.
- Demo ứng dụng: phần chung của nhóm.
- Tuning kiến trúc quá sâu hoặc neural architecture search.

## 4. Dữ liệu

### 4.1. Nguồn dữ liệu

| Dataset | Tổng ảnh | Số lớp gốc | Ảnh dùng cho 5 lớp | License |
|---|---:|---:|---:|---|
| RealWaste | 4.752 | 9 | 3.092 | CC BY 4.0 |
| TrashNet | 2.527 | 6 | 2.390 | MIT |
| **Tổng** | | | **5.482** | |

- RealWaste: ảnh rác thải trong môi trường thực tế, nền phức tạp.
- TrashNet: ảnh rác thải trên nền tương đối đơn giản.
- Sự khác biệt domain giữa hai nguồn là một phần của bài toán tổng quát hóa (RQ3),
  nhưng trong RQ1 cả hai được trộn theo cùng một split.

### 4.2. Mapping nhãn

| Lớp chuẩn | RealWaste | TrashNet |
|---|---|---|
| cardboard | Cardboard | cardboard |
| glass | Glass | glass |
| metal | Metal | metal |
| paper | Paper | paper |
| plastic | Plastic | plastic |

Các lớp bị loại khỏi RQ1:

- RealWaste: Food Organics, Miscellaneous Trash, Textile Trash, Vegetation.
- TrashNet: trash.

Lý do: không có lớp tương ứng đủ nhất quán giữa hai dataset, hoặc ngữ nghĩa khác
nhau. Tổng cộng 1.797 ảnh bị loại.

### 4.3. Phân bố ảnh

| Lớp | RealWaste | TrashNet | Tổng |
|---|---:|---:|---:|
| cardboard | 461 | 403 | 864 |
| glass | 420 | 501 | 921 |
| metal | 790 | 410 | 1.200 |
| paper | 500 | 594 | 1.094 |
| plastic | 921 | 482 | 1.403 |
| **Tổng** | **3.092** | **2.390** | **5.482** |

Dữ liệu mất cân bằng ở mức vừa phải: plastic nhiều nhất (1.403), cardboard ít nhất
(864). Vì vậy Macro-F1 là metric chính thay vì accuracy.

### 4.4. Split

- Version: `data/splits/v1/`.
- Tỷ lệ: 70% train / 15% validation / 15% test.
- Seed: 42.
- Stratified theo cặp `(dataset, class)`.
- Chống leakage bằng pHash: ảnh gần trùng được gom thành group; group không bị tách
  giữa các split.
- Ảnh trùng phát hiện: 35; số group: 5.447.

| Split | RealWaste | TrashNet | Tổng |
|---|---:|---:|---:|
| Train | 2.163 | 1.670 | 3.833 |
| Validation | 466 | 360 | 826 |
| Test | 463 | 360 | 823 |
| **Tổng** | **3.092** | **2.390** | **5.482** |

Mỗi lớp có ít nhất 130 ảnh ở validation và test.

### 4.5. Preprocessing

- Resize cạnh ngắn 256, CenterCrop 224 cho validation/test.
- Train dùng RandomResizedCrop 224 khi có augmentation; RQ1 tắt augmentation nên
  dùng Resize + CenterCrop.
- Normalize bằng mean/std ImageNet:
  - mean = 0.485, 0.456, 0.406
  - std = 0.229, 0.224, 0.225
- Không dùng augmentation trong RQ1 để cô lập biến kiến trúc/pretrained.
- Không dùng class weighting trong RQ1.

### 4.6. Kiểm soát leakage

- Test set chỉ dùng một lần sau khi khóa cấu hình.
- Validation dùng cho chọn epoch, chọn cấu hình và debug.
- Không dùng test để chọn threshold, epoch, learning rate hoặc biến thể.
- Split CSV được commit để local và GPU dùng đúng một phiên bản.

## 5. Kiến thức nền

### 5.1. Convolutional Neural Network

- Convolution 2D trích xuất đặc trưng cục bộ.
- Kernel 3×3, padding 1 giữ kích thước không gian.
- MaxPool giảm độ phân giải, tăng receptive field.
- BatchNorm ổn định phân phối activation, cho phép learning rate cao hơn.
- ReLU tạo phi tuyến, giảm vanishing gradient so với sigmoid/tanh.
- Global Average Pooling giảm số tham số và overfitting so với flatten + FC lớn.
- Dropout regularize lớp phân loại.

### 5.2. Transfer learning

- Pretrained trên ImageNet học đặc trưng cạnh, texture, hình dạng.
- Fine-tuning thay classifier cuối và huấn luyện lại một phần hoặc toàn bộ backbone.
- Freeze backbone giúp ổn định giai đoạn đầu, giảm chi phí tính toán.
- Unfreeze một số block cuối cho phép thích nghi đặc trưng domain mới.
- Learning rate backbone thường nhỏ hơn head để tránh phá pretrained features.

### 5.3. MobileNetV3_small

- Depthwise separable convolution: tách depthwise và pointwise để giảm FLOPs.
- Inverted residual block: mở rộng kênh, depthwise conv, chiếu lại.
- Squeeze-and-Excitation: học trọng số kênh.
- Hardswish/Hardsigmoid: activation rẻ hơn cho thiết bị tài nguyên hạn chế.
- MobileNetV3_small cân bằng giữa độ chính xác và chi phí, phù hợp demo/edge.

### 5.4. Tối ưu và huấn luyện

- AdamW: Adam có weight decay tách biệt, phù hợp CNN hiện đại.
- Cosine annealing: giảm learning rate mượt theo epoch.
- Mixed precision (AMP): tăng tốc trên GPU, giảm bộ nhớ.
- Early stopping: dừng khi validation không cải thiện.
- Checkpoint: lưu `last` để resume, `best` theo validation Macro-F1.
- Seed cố định: random, NumPy, PyTorch, CUDA.

### 5.5. Đánh giá phân loại

- Accuracy: tỷ lệ dự đoán đúng.
- Balanced accuracy: trung bình recall các lớp, phù hợp mất cân bằng.
- Precision/Recall/F1 macro: trung bình không trọng số theo lớp.
- Confusion matrix: phân bố lỗi giữa các lớp.
- Per-class F1: phát hiện lớp yếu.
- FLOPs: chi phí tính toán lý thuyết cho một ảnh.
- Latency: thời gian suy luận thực tế.
- VRAM: bộ nhớ GPU tối đa khi suy luận.

## 6. Kiến trúc mô hình

### 6.1. Custom CNN baseline

Thiết kế:

| Stage | Thành phần | Output |
|---|---|---|
| Stem | Conv 3×3 (3→32), BN, ReLU, MaxPool | 32×112×112 |
| Stage 1 | 2× [Conv 3×3 (32→64), BN, ReLU], MaxPool | 64×56×56 |
| Stage 2 | 2× [Conv 3×3 (64→128), BN, ReLU], MaxPool | 128×28×28 |
| Stage 3 | 2× [Conv 3×3 (128→256), BN, ReLU], MaxPool | 256×14×14 |
| Pool | AdaptiveAvgPool2d(1) | 256 |
| Head | Dropout 0.3 → Linear(256→5) | 5 logits |

Thông số:

- Channels: 32-64-128-256.
- Convs/stage: 1-2-2-2.
- Dropout: 0.3.
- Params: 1.174.501.
- FLOPs đo được: khoảng 5,17 GFLOPs cho một ảnh 224×224.
- Khởi tạo Kaiming He cho Conv, Xavier cho Linear.
- Không softmax; dùng CrossEntropyLoss.

Biến thể residual:

- Thêm skip connection cấp stage, có 1×1 conv khi đổi số kênh.
- Params: 1.218.405.
- FLOPs: khoảng 5,33 GFLOPs.
- Đã thử trong debug; không được chọn làm cấu hình cuối vì không cải thiện
  validation đủ rõ.

### 6.2. MobileNetV3_small

- Nguồn: torchvision, pretrained `IMAGENET1K_V1`.
- Thay classifier cuối `1024→1000` bằng `1024→5`.
- Dropout mặc định 0.2.
- Params: 1.522.981.
- FLOPs đo được: khoảng 0,111 GFLOPs cho một ảnh 224×224.
- Có hai chế độ:
  - Scratch: khởi tạo ngẫu nhiên, không pretrained.
  - Pretrained: nạp trọng số ImageNet, fine-tune.

Chiến lược fine-tune:

- Có thể full fine-tune từ epoch đầu, hoặc freeze backbone vài epoch rồi mở một số
  block cuối.
- Backbone dùng learning rate nhỏ hơn head (scale 0.1).
- Cấu hình cuối được chọn trên validation ở budget 50 epoch; tài liệu này sẽ cập nhật
  giá trị chốt cùng kết quả final.

### 6.3. So sánh quy mô

| Mô hình | Params | GFLOPs | Ghi chú |
|---|---:|---:|---|
| Custom CNN | 1.174.501 | 5,17 | Baseline từ đầu |
| Custom CNN residual | 1.218.405 | 5,33 | Biến thể debug |
| MobileNetV3_small | 1.522.981 | 0,11 | Scratch hoặc pretrained |

Điểm quan trọng: số tham số tương đương nhưng Custom CNN có FLOPs cao hơn khoảng
46 lần. Vì vậy RQ1 không thể chỉ so sánh accuracy; cần báo cáo cả chi phí tính toán.

## 7. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python 3.12 | Ngôn ngữ chính |
| PyTorch | Định nghĩa, huấn luyện, suy luận |
| torchvision | MobileNetV3, transform, pretrained weights |
| NumPy | Xử lý số, pHash |
| pandas | Tổng hợp kết quả, phân tích |
| scikit-learn | Precision, recall, F1, confusion matrix |
| matplotlib | Learning curves, confusion matrix, Grad-CAM |
| uv | Quản lý môi trường và dependency |
| pytest | Kiểm thử |
| ruff | Lint và format |
| Google Colab / GPU riêng | Huấn luyện |
| Google Drive | Lưu kết quả, bundle dữ liệu |
| Git / GitHub | Quản lý mã nguồn |
| Streamlit | Demo sau khi hoàn thành |

## 8. Thiết kế thực nghiệm

### 8.1. Ba nhánh

| Arm | Mô tả | Pretrained |
|---|---|---|
| C | Custom CNN từ đầu | Không |
| S | MobileNetV3_small từ đầu | Không |
| P | MobileNetV3_small ImageNet | Có |

### 8.2. Biến cố định

- Split v1.
- Input 224×224, normalize ImageNet.
- Không augmentation, không class weighting.
- Optimizer AdamW.
- Batch size 64.
- Weight decay 1e-4.
- Scheduler cosine.
- 50 epoch, early stopping theo validation Macro-F1.
- 5 seed: 42, 123, 2024, 7, 2025.
- Cùng code training, cùng metric, cùng quy trình đánh giá.

### 8.3. Biến thay đổi

- Kiến trúc: Custom CNN vs MobileNetV3_small.
- Pretrained: có vs không (chỉ MobileNet).
- Với MobileNet pretrained: chiến lược freeze/unfreeze được chọn trên validation.

### 8.4. Ma trận thí nghiệm

| Câu hỏi | Thí nghiệm | Metric | Tiêu chí |
|---|---|---|---|
| Transfer learning có tốt hơn? | C vs P | Macro-F1 | P > C ≥ 0.02, ≥ 4/5 seed |
| Kiến trúc đóng góp bao nhiêu? | C vs S | Macro-F1 | S > C nếu kiến trúc có lợi |
| Pretrained đóng góp bao nhiêu? | S vs P | Macro-F1 | P > S nếu pretrained có lợi |
| Chi phí đánh đổi? | C, S, P | Params, GFLOPs, latency | Báo cáo định lượng |

### 8.5. Chọn cấu hình Custom CNN

Đã thực hiện trên validation:

- 4 ứng viên channels/dropout.
- 6 thí nghiệm debug: budget, learning rate, batch size, weight decay.
- 1 biến thể residual.

Cấu hình chốt:

- Channels 32-64-128-256.
- Convs/stage 1-2-2-2.
- Dropout 0.3.
- 50 epoch.
- Learning rate 1e-3.
- Batch 64.
- Weight decay 1e-4.
- Không residual.

Chi tiết: `docs/RQ1/customcnn_debug.md` và `docs/RQ1/model_selection.md`.

### 8.6. Chọn cấu hình MobileNet pretrained

Đã sàng lọc trên validation:

- Full fine-tune.
- Freeze 3, mở 2 block.
- Freeze 3, dropout 0.3.
- Freeze 5, mở 3 block.

Hai cấu hình tốt nhất được xác nhận ở 50 epoch trước khi khóa.
Chi tiết: `docs/RQ1/mobilenet_selection.md`.

### 8.7. Chống leakage và HARKing

- Test chỉ đánh giá một lần sau khi khóa cấu hình.
- Mọi lựa chọn dùng validation.
- Plan được commit trước khi chạy.
- Không đổi tiêu chí sau khi xem test.
- Nếu giả thuyết sai, báo cáo trung thực và phân tích nguyên nhân.

## 9. Quy trình thực hiện

### 9.1. Chuẩn bị dữ liệu

- Tải RealWaste và TrashNet.
- Kiểm tra cấu trúc, số ảnh, license.
- Mapping nhãn về 5 lớp.
- Tính pHash, gom group, chia split v1.
- Commit split CSV và summary.
- Tạo bundle dữ liệu cho môi trường GPU.

### 9.2. Xây dựng mô hình

- Custom CNN trong `src/models/custom_cnn.py`.
- MobileNetV3_small trong `src/models/mobilenet.py`.
- Registry trong `src/models/factory.py`.
- Hàm freeze/unfreeze và param groups cho transfer learning.

### 9.3. Training engine

- `src/training/runner.py`.
- Train/validation loop.
- AMP khi có CUDA.
- Early stopping theo validation Macro-F1.
- Checkpoint `last` và `best`.
- Resume từ `last`.
- Ghi log JSON/JSONL.
- Chỉ tạo test loader khi `evaluate_test` bật.

### 9.4. Logging và tái lập

Mỗi run lưu:

- `run.json`: config, seed, git commit, trạng thái.
- `plan.yaml`: bản copy plan đã commit.
- `config.yaml`, `environment.json`.
- `history.jsonl`: metric từng epoch.
- `metrics.json`: kết quả validation/test.
- `checkpoints/last.pt`, `checkpoints/best.pt`.
- `predictions/test.csv`.

`environment.json` ghi Python, PyTorch, CUDA, GPU, phiên bản thư viện.

### 9.5. Chạy thí nghiệm

- Plan RQ1: `plans/rq1/model_comparison.yaml`.
- Config: `configs/rq1/`.
- Batch runner: `scripts/run_batch.py`.
- Tự skip run completed, resume run dở, export zip cuối batch.
- Kết quả có thể chạy trên Colab hoặc GPU riêng.

### 9.6. Phân tích

- `scripts/analyze_rq1.py`: bảng, curves, confusion matrix, lỗi.
- `scripts/measure_efficiency.py`: params, FLOPs, latency, VRAM.
- `scripts/gradcam.py`: Grad-CAM trên ảnh đúng/sai.
- `scripts/collect_results.py`: bảng tổng hợp.
- `docs/RQ1/rq1_results.md`: kết quả final (cập nhật sau).

## 10. Độ đo

### 10.1. Metric phân loại

- Accuracy.
- Balanced accuracy.
- Precision macro.
- Recall macro.
- Macro-F1.
- Per-class F1, precision, recall, support.
- Confusion matrix.

Macro-F1 là metric chính vì dữ liệu mất cân bằng và mỗi lớp quan trọng như nhau.

### 10.2. Metric hiệu năng

- Số tham số.
- GFLOPs cho một ảnh 224×224.
- Latency batch 1 và batch 32.
- Throughput ảnh/giây.
- VRAM tối đa khi suy luận.
- Thời gian huấn luyện một run.

### 10.3. Thống kê

- Mean ± std trên 5 seed.
- Paired difference theo seed.
- Khoảng tin cậy bootstrap cho chênh lệch Macro-F1 (nếu thực hiện).
- Không dùng p-value đơn lẻ để khẳng định khi chỉ có 5 seed.

### 10.4. Phân tích định tính

- Confusion matrix: cặp lớp bị nhầm.
- Ảnh sai tiêu biểu.
- Grad-CAM: vùng ảnh ảnh hưởng đến dự đoán.
- So sánh lỗi giữa Custom CNN và MobileNet.
- Phân tích theo nguồn: RealWaste vs TrashNet.

## 11. Phương pháp đánh giá

### 11.1. Chọn epoch

- Theo validation Macro-F1.
- Không dùng test.
- Early stopping patience 7–10.
- Lưu best checkpoint tại epoch tốt nhất.

### 11.2. So sánh ba arm

- Cùng split, cùng budget, cùng seed.
- Báo cáo mean ± std.
- So sánh từng seed.
- Nếu chênh lệch nhỏ hơn biến động giữa seed, kết luận là “không đủ bằng chứng
  khác biệt” thay vì khẳng định.

### 11.3. Tiêu chí trả lời RQ1

- Nếu P > C ≥ 0.02 và ≥ 4/5 seed: ủng hộ giả thuyết.
- Nếu P > C nhưng < 0.02: cải thiện nhỏ, cần báo cáo trung thực.
- Nếu P ≈ C: transfer learning không tạo khác biệt thực chất trên dữ liệu này.
- Nếu P < C: giả thuyết bị bác bỏ, phân tích nguyên nhân.

### 11.4. Phân rã kiến trúc và pretrained

- So sánh C, S, P.
- Nếu S > C: kiến trúc đóng góp.
- Nếu P > S: pretrained đóng góp.
- Nếu P ≈ S: pretrained không tạo khác biệt.
- Báo cáo cả chi phí tính toán để tránh kết luận “P tốt hơn” mà bỏ qua FLOPs.

### 11.5. Phân tích lỗi

- Confusion matrix từng arm.
- Per-class F1.
- Top cặp nhầm.
- Ảnh sai có nền phức tạp hay không.
- Grad-CAM xem mô hình tập trung vào vật thể hay background.
- So sánh lỗi giữa Custom CNN và MobileNet.

### 11.6. Đánh giá hiệu năng

- Accuracy/Macro-F1 trên mỗi GFLOPs.
- Latency trên CPU và GPU.
- VRAM.
- Kết luận về đánh đổi giữa chất lượng và chi phí.

### 11.7. External validation trên GD

Sau khi RQ1 final khóa, ba arm được đánh giá thêm trên GD — nguồn không dùng để train.

- 7.010 ảnh GD mapping về 5 lớp.
- Loại 2.200 ảnh trùng với split v1 và 4 ảnh trùng nội bộ.
- Còn 4.806 ảnh external test.
- Không dùng external để chọn model.
- Kết quả là mục “External validation”, không thay đổi kết luận chính.

Chi tiết: `docs/RQ1/external_validation.md`.

## 12. Trạng thái và kế hoạch

### 12.1. Đã hoàn thành

- Data pipeline, split v1, label mapping.
- Custom CNN và MobileNetV3_small.
- Training engine, logging, checkpoint, resume.
- Chọn cấu hình Custom CNN trên validation.
- Debug Custom CNN.
- Sàng lọc MobileNet trên validation.
- Plan RQ1, ma trận thí nghiệm.
- Công cụ phân tích, efficiency, Grad-CAM.

### 12.2. Chưa hoàn thành

- Xác nhận MobileNet ở 50 epoch.
- Chốt cấu hình MobileNet pretrained.
- Chạy final 3 arm × 5 seed.
- Đánh giá test một lần.
- Bảng kết quả final.
- Hình confusion matrix, curves, Grad-CAM final.
- Đo efficiency final.
- Viết kết luận RQ1.

### 12.3. Kế hoạch chạy

1. Xác nhận MobileNet 50 epoch trên validation.
2. Chốt cấu hình MobileNet.
3. Chạy Custom CNN, MobileNet pretrained, MobileNet scratch với 5 seed.
4. Đánh giá test một lần.
5. Tổng hợp bảng, hình, phân tích lỗi.
6. Đo efficiency và Grad-CAM.
7. Cập nhật tài liệu và báo cáo.

### 12.4. Kết quả final

Phần này sẽ được cập nhật sau khi chạy xong:

- Bảng Macro-F1 mean ± std cho 3 arm.
- Accuracy, balanced accuracy, precision/recall macro.
- Per-class F1.
- Confusion matrix.
- Params, GFLOPs, latency, VRAM.
- Hình learning curves.
- Hình Grad-CAM.
- Kết luận RQ1.

Không đưa số liệu validation vào phần kết quả final.

## 13. Rủi ro và hạn chế

### 13.1. Dữ liệu

- Chỉ 5.482 ảnh cho 5 lớp, tương đối nhỏ cho deep learning.
- Mất cân bằng lớp.
- Khác biệt domain giữa RealWaste và TrashNet.
- Ảnh có thể có background không liên quan.
- Nhãn có thể có nhiễu ở một số ảnh.

### 13.2. Thiết kế

- So sánh kiến trúc khác nhau về FLOPs, không chỉ số tham số.
- Pretrained ImageNet khác domain rác thải.
- Không augmentation ở RQ1 là lựa chọn có chủ đích, nhưng có thể làm cả hai mô hình
  overfit.
- Chỉ 5 seed; độ tin cậy thống kê có hạn.
- Test set chỉ đánh giá một lần; không thể kiểm tra lại nếu phát hiện lỗi sau đó.

### 13.3. Kỹ thuật

- Khác biệt PyTorch/CUDA giữa local và Colab/GPU có thể ảnh hưởng nhỏ đến kết quả.
- AMP có thể gây khác biệt số học.
- DataLoader và thứ tự ảnh có thể ảnh hưởng đến seed.
- Resume có thể làm lệch nhẹ lịch sử huấn luyện.

### 13.4. Cách giảm thiểu

- Cố định seed và ghi environment.
- Dùng cùng split, cùng code, cùng metric.
- Test chỉ một lần.
- Báo cáo trung thực cả kết quả bất lợi.
- Lưu plan và log để tái lập.

## 14. Tài liệu tham khảo

1. Howard, A. et al. (2019). Searching for MobileNetV3. ICCV.
2. Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste: A Novel Real-Life Data
   Set for Landfill Waste Classification Using Deep Learning. Information.
3. Thung, G., & Yang, M. (2016). Classification of Trash for Recyclability Status.
   Stanford University.
4. Deng, J. et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. CVPR.
5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
6. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
   via Gradient-based Localization. ICCV.
7. PyTorch documentation.
8. torchvision documentation.
9. scikit-learn documentation.
10. UCI Machine Learning Repository — RealWaste dataset.
11. GitHub — TrashNet dataset.

## 15. Phụ lục

### 15.1. File liên quan

| File | Nội dung |
|---|---|
| `docs/RQ1/rq1_debug.md` | Debug Custom CNN |
| `docs/RQ1/customcnn_debug.md` | Báo cáo debug sinh tự động |
| `docs/RQ1/model_selection.md` | Chọn cấu hình Custom CNN |
| `docs/RQ1/mobilenet_selection.md` | Chọn cấu hình MobileNet |
| `docs/RQ1/rq1_results.md` | Kết quả final (cập nhật sau) |
| `docs/RQ1/rq1_efficiency.md` | Params, FLOPs, latency |
| `docs/training.md` | Hướng dẫn chạy |
| `docs/models.md` | Kiến trúc mô hình |
| `docs/splits.md` | Split v1 |
| `plans/rq1/model_comparison.yaml` | Plan RQ1 |
| `configs/rq1/` | Config thực nghiệm |

### 15.2. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| RQ | Research Question |
| Macro-F1 | Trung bình F1 không trọng số theo lớp |
| Transfer learning | Dùng pretrained weights rồi fine-tune |
| Scratch | Huấn luyện từ đầu, không pretrained |
| Freeze | Đóng băng tham số backbone |
| Unfreeze | Mở một số block cuối để fine-tune |
| FLOPs | Số phép tính dấu phẩy động cho một lần forward |
| AMP | Automatic Mixed Precision |
| Grad-CAM | Bản đồ nhiệt vùng ảnh ảnh hưởng đến dự đoán |
| HARKing | Đặt giả thuyết sau khi đã xem kết quả |

### 15.3. Ghi chú cập nhật

- Tài liệu này mô tả thiết kế và quy trình, không chứa kết quả final.
- Khi RQ1 chạy xong, cập nhật mục 12.4 và liên kết `docs/RQ1/rq1_results.md`.
- Nếu cấu hình MobileNet cuối thay đổi, cập nhật mục 6.2 và 8.6.
