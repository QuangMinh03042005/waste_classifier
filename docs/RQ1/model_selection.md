# Chọn cấu hình Custom CNN

Bước chọn cấu hình chỉ dùng **validation**, không đánh giá test (`evaluate_test: false`).

## Ứng viên

| ID | Channels | Conv/stage | Dropout | Epochs | Config |
| --- | --- | --- | --- | --- | --- |
| cand_a | [24, 48, 96, 192] | [1, 2, 2, 2] | 0.3000 | 12 | customcnn_cand_a.yaml |
| cand_b | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 12 | customcnn_cand_b.yaml |
| cand_c | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.5000 | 12 | customcnn_cand_c.yaml |
| cand_d | [48, 96, 192, 384] | [1, 2, 2, 2] | 0.3000 | 12 | customcnn_cand_d.yaml |

## Quy tắc chọn

- Sàng lọc seed 42 cho cả 4 ứng viên.
- Chọn top-2 theo val Macro-F1, chạy thêm seed 123 và 2024.
- Chọn theo mean val Macro-F1; nếu chênh < 0.005, ưu tiên ít tham số/FLOPs hơn.
- Không dùng test ở bất kỳ bước nào.

## Kết quả

| ID | Seed | Val Macro-F1 | Val Loss | Best epoch | Params | Duration (s) | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cand_a | 42 | 0.8104 | 0.5672 | 12 | 661421 | 325.3200 | 4d32059a |
| cand_b | 123 | 0.8148 | 0.5723 | 11 | 1174501 | 337.5780 | 4d32059a |
| cand_b | 2024 | 0.8255 | 0.5359 | 12 | 1174501 | 338.4360 | 4d32059a |
| cand_b | 42 | 0.8218 | 0.5450 | 12 | 1174501 | 343.4760 | 4d32059a |
| cand_c | 42 | 0.8135 | 0.5554 | 12 | 1174501 | 342.8160 | 4d32059a |
| cand_d | 123 | 0.8034 | 0.5904 | 12 | 2639573 | 380.5180 | 4d32059a |
| cand_d | 2024 | 0.8140 | 0.5676 | 11 | 2639573 | 384.2230 | 4d32059a |
| cand_d | 42 | 0.8175 | 0.5673 | 12 | 2639573 | 383.4200 | 4d32059a |

## Tổng hợp theo ứng viên

| ID | Số seed | Val Macro-F1 mean | Std | Params |
| --- | --- | --- | --- | --- |
| cand_a | 1 | 0.8104 |  | 661421 |
| cand_b | 3 | 0.8207 | 0.0055 | 1174501 |
| cand_c | 1 | 0.8135 |  | 1174501 |
| cand_d | 3 | 0.8116 | 0.0073 | 2639573 |

## Kết luận

- Chọn **cand_b** với val Macro-F1 trung bình **0.8207** (std 0.0055), 1174501 tham số.
- Đây là kết quả chọn trên validation; test chưa được đánh giá.
- Dùng cấu hình này cho RQ1 với 30 epoch × 3 seed ở bước tiếp theo.
