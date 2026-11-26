# Ma trận thí nghiệm

- Tạo lúc: `2026-11-26T04:47:26.932384+00:00`
- Plan core: 4
- Experiment core: 11
- Plan pilot/selection: 3
- Experiment pilot/selection: 17

## Ma trận core

### RQ1 – Model Architecture

### `rq1_model_comparison` - So sánh Custom CNN, MobileNetV3_small scratch và pretrained

- **Câu hỏi:** Transfer learning với MobileNetV3_small có cải thiện Macro-F1 so với Custom CNN huấn luyện từ đầu trên cùng split dữ liệu không; phần cải thiện đến từ kiến trúc hay từ pretrained weights?
- **Giả thuyết:** MobileNetV3_small pretrained đạt Macro-F1 cao hơn Custom CNN ít nhất 0.02 và cao hơn MobileNetV3_small from scratch; khoảng cách với scratch lớn hơn khoảng cách giữa scratch và Custom CNN.
- **Kết quả kỳ vọng:** Pretrained > scratch > Custom CNN. Chênh lệch rõ nhất ở Glass và Paper, nơi đặc trưng texture quan trọng và pretrained features có lợi.
- **Nếu giả thuyết sai:** Nếu pretrained không cải thiện, hoặc scratch ≈ pretrained, báo cáo trung thực và phân tích nguyên nhân (domain shift, kích thước dữ liệu, overfitting). Không đổi tiêu chí hoặc chọn seed sau khi xem test.
- **Metric:** macro_f1
- **Test policy:** Chọn epoch theo val Macro-F1; test chỉ đánh giá một lần sau khi khóa cấu hình.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/model_comparison.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn | baseline | - | - | macro_f1 | Pretrained cao hơn Custom CNN >= 0.02 và cao hơn ở >= 4/5 seed. So sánh thêm với scratch để kết luận phần đóng góp của kiến trúc và pretrained weights. | 42, 123, 2024, 7, 3407 | active |
| rq1_mobilenetv3s_full | variant | Thay Custom CNN bằng MobileNetV3_small pretrained ImageNet. | split v1, stratified theo (dataset, class), chống trùng bằng pHash; input 224x224, normalize mean/std ImageNet; không augmentation, không class weighting; budget epoch giống Custom CNN (30 hoặc 50 theo kết quả debug); AdamW, batch 64, early stopping patience 7; seed 42, 123, 2024, 7, 3407 | macro_f1 | Pretrained cao hơn Custom CNN >= 0.02 và cao hơn ở >= 4/5 seed. So sánh thêm với scratch để kết luận phần đóng góp của kiến trúc và pretrained weights. | 42, 123, 2024, 7, 3407 | active |
| rq1_mobilenetv3s_scratch | variant | MobileNetV3_small from scratch để tách ảnh hưởng của kiến trúc khỏi pretrained weights. | split v1, input 224x224, không augmentation, không class weighting; không freeze, dropout 0.2, AdamW LR 1e-3, batch 64; budget epoch giống Custom CNN (30 hoặc 50); seed 42, 123, 2024, 7, 3407 | macro_f1 | Pretrained cao hơn Custom CNN >= 0.02 và cao hơn ở >= 4/5 seed. So sánh thêm với scratch để kết luận phần đóng góp của kiến trúc và pretrained weights. | 42, 123, 2024, 7, 3407 | active |

### `rq1_scratch_fix` - Sửa recipe huấn luyện MobileNetV3_small scratch (validation-only)

- **Câu hỏi:** Recipe nào giúp MobileNetV3_small from scratch hội tụ ổn định trên split v1, sau khi sửa lỗi backbone learning rate bị scale như fine-tuning?
- **Giả thuyết:** Với backbone_lr_scale = 1.0 và warmup 3 epoch, scratch sẽ học được đặc trưng thay vì chỉ head memorize; LR 3e-4 ổn định hơn 1e-3.
- **Kết quả kỳ vọng:** Cả S1 và S2 đều vượt mức random; S2 có val Macro-F1 cao hơn hoặc ổn định hơn.
- **Nếu giả thuyết sai:** Nếu cả hai vẫn gần random, kiểm tra lại gradient/BN và ghi nhận scratch không hội tụ với recipe không augmentation. Không dùng test để chọn recipe.
- **Metric:** val_macro_f1
- **Test policy:** Không đánh giá test trong screening; recipe chốt trên validation.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/scratch_fix.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_scratch_s1 | baseline | - | - | val_macro_f1 | Chọn recipe có val Macro-F1 cao nhất và không gần random; nếu chênh < 0.005, chọn recipe ổn định hơn theo learning curves. | 42, 123, 2024, 7, 3407 | active |
| rq1_scratch_s2 | variant | Giảm LR xuống 3e-4, giữ backbone_lr_scale=1.0 và warmup 3 epoch. | split v1, input 224x224, không augmentation, không class weighting; MobileNetV3_small scratch, dropout 0.2, AdamW, batch 64; 50 epoch, early stopping patience 7, seed 42 cho screening; top-2 chạy thêm seed 123, 2024; final 5 seed nếu được chọn | val_macro_f1 | Chọn recipe có val Macro-F1 cao nhất và không gần random; nếu chênh < 0.005, chọn recipe ổn định hơn theo learning curves. | 42, 123, 2024, 7, 3407 | active |

### RQ2 – Data Augmentation & Class Weighting

### `rq2_augmentation_class_weighting` - Ablation Data Augmentation và Class Weighting

- **Câu hỏi:** Data Augmentation và Class Weighting đóng góp bao nhiêu vào Macro-F1 của MobileNetV3_small trên bài toán 5 lớp?
- **Giả thuyết:** Full model (augmentation + class weighting) đạt Macro-F1 cao hơn từng ablation; class weighting cải thiện recall của lớp thiểu số (Glass, Cardboard).
- **Kết quả kỳ vọng:** Full cao hơn no-augmentation rõ rệt về Macro-F1; class weighting tăng recall lớp thiểu số nhưng có thể giảm nhẹ accuracy toàn cục.
- **Nếu giả thuyết sai:** Nếu ablation không cải thiện, ghi nhận và phân tích theo lớp cùng đường cong train/validation. Không chọn biến thể dựa trên test set.
- **Metric:** macro_f1
- **Test policy:** Chọn epoch theo val Macro-F1; test chỉ đánh giá một lần cho mỗi run (mỗi biến thể x seed) và không dùng test để chọn epoch hay cấu hình.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq2/augmentation_ablation.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq2_mobilenetv3s_full | baseline | - | - | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |
| rq2_mobilenetv3s_noaug | variant | Bỏ Data Augmentation (train transform chỉ còn Resize 256 + CenterCrop 224), giữ nguyên class weighting. | split v1, MobileNetV3_small pretrained ImageNet, full fine-tune, dropout 0.2; class weighting w_c = N/(K*n_c), clip [0.5, 3.0]; weight trên train v1: cardboard 1.271, glass 1.189, metal 0.914, paper 1.001, plastic 0.782 (clip không kích hoạt); input 224x224, chuẩn hóa mean/std ImageNet; 50 epoch cho baseline và mọi biến thể, batch 64, AdamW lr 1e-3, weight decay 1e-4, cosine, early stopping patience 7 theo val Macro-F1; seed 42, 123, 2024 | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |
| rq2_mobilenetv3s_nocw | variant | Bỏ Class Weighting (CrossEntropyLoss không trọng số), giữ nguyên Data Augmentation. | split v1, MobileNetV3_small pretrained ImageNet, full fine-tune, dropout 0.2; augmentation: RandomResizedCrop 224 scale 0.7-1.0, RandomHorizontalFlip p=0.5, RandomRotation 15 độ, ColorJitter 0.2, RandomErasing p=0.1; input 224x224, chuẩn hóa mean/std ImageNet; 50 epoch cho baseline và mọi biến thể, batch 64, AdamW lr 1e-3, weight decay 1e-4, cosine, early stopping patience 7 theo val Macro-F1; seed 42, 123, 2024 | macro_f1 | Full cao hơn mỗi ablation >= 0.01 Macro-F1 (trung bình 3 seed). | 42, 123, 2024 | active |

### RQ3 – Generalization

### `rq3_cross_dataset_generalization` - Tổng quát hóa chéo nguồn dữ liệu

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
| rq3_train_realwaste_test_trashnet | baseline | - | - | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |
| rq3_train_trashnet_test_realwaste | variant | Đảo chiều nguồn: huấn luyện trên TrashNet, kiểm thử toàn bộ RealWaste. | 5 lớp chung, input 224x224, MobileNetV3_small pretrained ImageNet; không augmentation, không class weighting; AdamW, batch 64, tối đa 30 epoch, early stopping patience 7; seed 42, 123, 2024 | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |
| rq3_train_pooled_test_each | variant | Huấn luyện trên dữ liệu kết hợp 2 nguồn, kiểm thử riêng từng nguồn. | split v1 cho phần train/val; test của mỗi nguồn giữ nguyên; 5 lớp chung, input 224x224, MobileNetV3_small pretrained ImageNet; không augmentation, không class weighting; AdamW, batch 64, tối đa 30 epoch, early stopping patience 7; seed 42, 123, 2024 | macro_f1 | Pooled cao hơn single-source >= 0.02 Macro-F1 (trung bình 3 seed) trên cả hai test set RealWaste và TrashNet. | 42, 123, 2024 | active |

## Pilot / Selection (completed)

Các plan này dùng validation để chọn cấu hình hoặc debug; không nằm trong kết luận chính.

### RQ1 – Model Architecture

### `rq1_custom_cnn_debug` - Debug và chọn cấu hình Custom CNN (validation-only)

- **Câu hỏi:** Custom CNN B có đang underfit/overfit hoặc gặp vấn đề optimization không, và cấu hình nào (budget, LR, batch size, weight decay, residual) cho val Macro-F1 tốt nhất?
- **Giả thuyết:** Cấu hình B hiện tại chưa hội tụ ở 12 epoch; tăng lên 30-50 epoch sẽ cải thiện rõ. Residual có thể tăng val Macro-F1 nhưng không vượt quá 0.01.
- **Kết quả kỳ vọng:** D1 (50 epoch) cao hơn D0; D2/D3 cho thấy LR 1e-3 gần tối ưu; D4/D5 thay đổi nhỏ; R1 residual không vượt trội rõ nếu dữ liệu chỉ ~5.5k ảnh.
- **Nếu giả thuyết sai:** Nếu overfit 1 batch thất bại, dừng và sửa pipeline. Nếu val vẫn thấp sau 50 epoch, kiểm tra lại dữ liệu/transform và ghi nhận giới hạn. Không dùng test.
- **Metric:** val_macro_f1
- **Test policy:** Không đánh giá test trong bước debug.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/custom_cnn_debug.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn_d0 | baseline | - | - | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_d1 | variant | Tăng budget lên 50 epoch. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; AdamW LR 1e-3, batch 64, weight decay 1e-4, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_d2 | variant | Giảm learning rate xuống 3e-4. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; 30 epoch, batch 64, weight decay 1e-4, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_d3 | variant | Tăng learning rate lên 3e-3. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; 30 epoch, batch 64, weight decay 1e-4, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_d4 | variant | Giảm batch size xuống 32. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; 30 epoch, AdamW LR 1e-3, weight decay 1e-4, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_d5 | variant | Tăng weight decay lên 1e-3. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; 30 epoch, AdamW LR 1e-3, batch 64, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |
| rq1_customcnn_r1 | variant | Thêm residual connection ở mỗi stage. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2, dropout 0.3; 30 epoch, AdamW LR 1e-3, batch 64, weight decay 1e-4, seed 42 | val_macro_f1 | D1 > D0 thêm >= 0.01 thì final dùng 50 epoch. Nếu R1 > D0 thêm >= 0.01 và ổn định thì dùng residual. Chỉ thay một yếu tố mỗi biến thể. | 42 | completed |

### `rq1_custom_cnn_selection` - Chọn cấu hình Custom CNN baseline (validation-only)

- **Câu hỏi:** Cấu hình Custom CNN nào cho cân bằng tốt nhất giữa val Macro-F1, số tham số và chi phí tính toán trên split v1?
- **Giả thuyết:** Cấu hình B (32-64-128-256, dropout 0.3) có val Macro-F1 cao nhất hoặc tương đương cấu hình D nhưng ít tham số/FLOPs hơn.
- **Kết quả kỳ vọng:** A underfit nhẹ; B và D tương đương về val Macro-F1; C (dropout 0.5) hội tụ chậm hơn. Chọn B nếu chênh lệch val Macro-F1 so với D nhỏ hơn 0.005.
- **Nếu giả thuyết sai:** Nếu mọi ứng viên có val Macro-F1 < 0.60, dừng và kiểm tra lại dữ liệu, learning rate, augmentation trước khi chốt baseline. Không đánh giá test để chọn cấu hình.
- **Metric:** val_macro_f1
- **Test policy:** Không đánh giá test trong bước chọn cấu hình (evaluate_test=false).
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/custom_cnn_selection.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_customcnn_cand_b | baseline | - | - | val_macro_f1 | Chọn cấu hình có mean val Macro-F1 cao nhất trên top-2 × 3 seed; nếu chênh < 0.005, chọn cấu hình ít tham số/FLOPs hơn và ít overfit hơn. | 42, 123, 2024 | completed |
| rq1_customcnn_cand_a | variant | Giảm kênh còn 24-48-96-192 để kiểm tra capacity tối thiểu. | split v1, input 224x224, không augmentation, không class weighting; convs/stage 1-2-2-2, AdamW lr 1e-3, batch 64, 12 epoch; seed 42 cho sàng lọc; top-2 chạy thêm seed 123, 2024 | val_macro_f1 | Chọn cấu hình có mean val Macro-F1 cao nhất trên top-2 × 3 seed; nếu chênh < 0.005, chọn cấu hình ít tham số/FLOPs hơn và ít overfit hơn. | 42, 123, 2024 | completed |
| rq1_customcnn_cand_c | variant | Tăng dropout từ 0.3 lên 0.5. | split v1, input 224x224, không augmentation, không class weighting; channels 32-64-128-256, convs/stage 1-2-2-2; AdamW lr 1e-3, batch 64, 12 epoch, seed 42 | val_macro_f1 | Chọn cấu hình có mean val Macro-F1 cao nhất trên top-2 × 3 seed; nếu chênh < 0.005, chọn cấu hình ít tham số/FLOPs hơn và ít overfit hơn. | 42, 123, 2024 | completed |
| rq1_customcnn_cand_d | variant | Tăng kênh lên 48-96-192-384 để kiểm tra capacity cao hơn. | split v1, input 224x224, không augmentation, không class weighting; convs/stage 1-2-2-2, dropout 0.3; AdamW lr 1e-3, batch 64, 12 epoch, seed 42 | val_macro_f1 | Chọn cấu hình có mean val Macro-F1 cao nhất trên top-2 × 3 seed; nếu chênh < 0.005, chọn cấu hình ít tham số/FLOPs hơn và ít overfit hơn. | 42, 123, 2024 | completed |

### `rq1_mobilenet_selection` - Chọn cấu hình fine-tune MobileNetV3_small (validation-only)

- **Câu hỏi:** Chiến lược freeze/unfreeze và dropout nào cho MobileNetV3_small pretrained hiệu quả nhất trên validation, với cùng split và ngân sách?
- **Giả thuyết:** M2 (freeze 3, unfreeze 2) hoặc M1 (full fine-tune) đạt val Macro-F1 cao nhất; M4 freeze quá lâu sẽ underfit trong 12 epoch.
- **Kết quả kỳ vọng:** M1/M2 tương đương; M3 (dropout 0.3) thấp hơn nhẹ; M4 thấp hơn do chỉ fine-tune 7 epoch.
- **Nếu giả thuyết sai:** Nếu top-2 vẫn tăng ở epoch 12, chạy tiếp 30 epoch trước khi kết luận. Không dùng test.
- **Metric:** val_macro_f1
- **Test policy:** Không đánh giá test trong bước chọn cấu hình.
- **Dữ liệu:** train = realwaste, trashnet; split = `v1`; lớp = cardboard, glass, metal, paper, plastic
- **File plan:** `plans/rq1/mobilenet_selection.yaml`

| Experiment | Loại | Thay đổi | Giữ cố định | Metric | Tiêu chí kết luận | Seeds | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_mobilenet_m2_freeze3 | baseline | - | - | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
| rq1_mobilenet_m1_full_ft | variant | Bỏ freeze, full fine-tune từ epoch 1. | split v1, input 224x224, không augmentation, không class weighting; pretrained ImageNet, dropout 0.2, AdamW LR 1e-3, batch 64; 12 epoch, seed 42; top-2 chạy thêm seed 123, 2024 | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
| rq1_mobilenet_m3_dropout03 | variant | Tăng dropout classifier lên 0.3. | split v1, input 224x224, không augmentation, không class weighting; pretrained ImageNet, freeze 3 epoch, mở 2 block cuối; AdamW LR 1e-3, batch 64, 12 epoch, seed 42 | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
| rq1_mobilenet_m4_freeze5 | variant | Freeze 5 epoch, mở 3 block cuối. | split v1, input 224x224, không augmentation, không class weighting; pretrained ImageNet, dropout 0.2; AdamW LR 1e-3, batch 64, 12 epoch, seed 42 | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
| rq1_mobilenet_m1_full_ft_50 | variant | Xác nhận M1 (full fine-tune) ở final budget 50 epoch. | split v1, input 224x224, không augmentation, không class weighting; pretrained ImageNet, dropout 0.2, không freeze; AdamW LR 1e-3, batch 64, seed 42, 123, 2024 | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
| rq1_mobilenet_m4_freeze5_50 | variant | Xác nhận M4 (freeze 5, mở 3 block) ở final budget 50 epoch. | split v1, input 224x224, không augmentation, không class weighting; pretrained ImageNet, dropout 0.2, freeze 5 epoch; AdamW LR 1e-3, batch 64, seed 42, 123, 2024 | val_macro_f1 | Chọn mean val Macro-F1 cao nhất; nếu chênh < 0.005, ưu tiên cấu hình ít trainable params/thời gian hơn. Vì final budget là 50 epoch, xác nhận M1 và M4 ở 50 epoch trước khi khóa cấu hình. | 42, 123, 2024 | completed |
