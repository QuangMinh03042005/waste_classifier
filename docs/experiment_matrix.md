# Ma trận thí nghiệm

- Tạo lúc: `2026-11-13T07:03:32.764575+00:00`
- Số plan: 3
- Số experiment: 8

## RQ1 – Model Architecture

### `rq1_model_comparison` — So sánh Custom CNN và MobileNetV3_small

- **Câu hỏi:** Transfer learning với MobileNetV3_small có cải thiện Macro-F1 so với Custom CNN huấn luyện từ đầu trên cùng split dữ liệu không?
- **Giả thuyết:** MobileNetV3_small pretrained ImageNet đạt Macro-F1 cao hơn Custom CNN ít nhất 0.02 và ổn định trên cả 3 seed.
- **Kết quả kỳ vọng:** MobileNetV3_small cao hơn baseline ở cả 3 seed; chênh lệch rõ nhất ở các lớp Glass và Paper, nơi đặc trưng texture/kết cấu quan trọng.
- **Nếu giả thuyết sai:** Nếu không cải thiện, báo cáo trung thực kết quả, phân tích overfitting/underfitting và chi phí tính toán. Không thay đổi tiêu chí hoặc chọn seed sau khi xem test.
- **Metric:** macro_f1
- **Test policy:** Chọn epoch theo val Macro-F1; test chỉ đánh giá một lần sau khi khóa cấu hình.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/model_comparison.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn | baseline | — | — | macro_f1 | Macro-F1 trung bình cao hơn baseline >= 0.02 và cao hơn ở >= 2/3 seed. | 42, 123, 2024 | active |
| rq1_mobilenetv3s_full | variant | Thay Custom CNN bằng MobileNetV3_small pretrained ImageNet. | split v1, stratified theo (dataset, class), chống trùng bằng pHash; input 224x224, normalize mean/std ImageNet; không augmentation, không class weighting; AdamW, batch 64, tối đa 30 epoch, early stopping patience 7; seed 42, 123, 2024 | macro_f1 | Macro-F1 trung bình cao hơn baseline >= 0.02 và cao hơn ở >= 2/3 seed. | 42, 123, 2024 | active |

## RQ2 – Data Augmentation & Class Weighting

### `rq2_augmentation_class_weighting` — Ablation Data Augmentation và Class Weighting

- **Câu hỏi:** Data Augmentation và Class Weighting đóng góp bao nhiêu vào Macro-F1 của MobileNetV3_small trên bài toán 5 lớp?
- **Giả thuyết:** Full model (augmentation + class weighting) đạt Macro-F1 cao hơn từng ablation; class weighting cải thiện recall của lớp thiểu số (Glass, Cardboard).
- **Kết quả kỳ vọng:** Full cao hơn no-augmentation rõ rệt về Macro-F1; class weighting tăng recall lớp thiểu số nhưng có thể giảm nhẹ accuracy toàn cục.
- **Nếu giả thuyết sai:** Nếu ablation không cải thiện, ghi nhận và phân tích theo lớp cùng đường cong train/validation. Không chọn biến thể dựa trên test set.
- **Metric:** macro_f1
- **Test policy:** Chọn epoch theo val Macro-F1; test chỉ đánh giá một lần cho mỗi biến thể.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq2/augmentation_ablation.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq2_mobilenetv3s_full | baseline | — | — | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |
| rq2_mobilenetv3s_noaug | variant | Bỏ Data Augmentation, giữ nguyên class weighting. | split v1, MobileNetV3_small pretrained ImageNet; class weighting theo tần suất lớp, clip [0.5, 3.0]; input 224x224, AdamW, batch 64, tối đa 30 epoch; seed 42, 123, 2024 | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |
| rq2_mobilenetv3s_nocw | variant | Bỏ Class Weighting, giữ nguyên Data Augmentation. | split v1, MobileNetV3_small pretrained ImageNet; augmentation: RandomResizedCrop, HFlip, rotation, color jitter; input 224x224, AdamW, batch 64, tối đa 30 epoch; seed 42, 123, 2024 | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |

## RQ3 – Generalization

### `rq3_cross_dataset_generalization` — Tổng quát hóa chéo nguồn dữ liệu

- **Câu hỏi:** Việc kết hợp dữ liệu từ RealWaste và TrashNet có cải thiện khả năng tổng quát hóa trên nguồn dữ liệu còn lại so với huấn luyện đơn nguồn không?
- **Giả thuyết:** Huấn luyện trên dữ liệu kết hợp cải thiện Macro-F1 cross-dataset so với huấn luyện chỉ trên một nguồn, đặc biệt khi kiểm thử trên TrashNet.
- **Kết quả kỳ vọng:** Pooled model cao hơn single-source trên cả hai test set ngoài miền; mức cải thiện lớn hơn ở TrashNet do nền ảnh đơn giản.
- **Nếu giả thuyết sai:** Nếu pooled không cải thiện, phân tích domain shift, phân phối lớp và chất lượng ảnh. Báo cáo cả trường hợp giả thuyết bị bác bỏ.
- **Metric:** macro_f1
- **Test policy:** Test ngoài miền chỉ đánh giá một lần; không dùng để chọn epoch.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq3/cross_dataset_generalization.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq3_train_realwaste_test_trashnet | baseline | — | — | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |
| rq3_train_trashnet_test_realwaste | variant | Đảo chiều nguồn: huấn luyện trên TrashNet, kiểm thử toàn bộ RealWaste. | 5 lớp chung, input 224x224, MobileNetV3_small pretrained ImageNet; không augmentation, không class weighting; AdamW, batch 64, tối đa 30 epoch, early stopping patience 7; seed 42, 123, 2024 | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |
| rq3_train_pooled_test_each | variant | Huấn luyện trên dữ liệu kết hợp 2 nguồn, kiểm thử riêng từng nguồn. | split v1 cho phần train/val; test của mỗi nguồn giữ nguyên; 5 lớp chung, input 224x224, MobileNetV3_small pretrained ImageNet; không augmentation, không class weighting; AdamW, batch 64, tối đa 30 epoch, early stopping patience 7; seed 42, 123, 2024 | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |
