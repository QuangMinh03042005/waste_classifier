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

_Chưa có run completed. Chạy các config trong `configs/rq1/selection/`._
