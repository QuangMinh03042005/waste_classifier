# Debug Custom CNN

Bước chọn cấu hình chỉ dùng **validation**, không đánh giá test (`evaluate_test: false`).

## Ứng viên

| ID | Channels | Conv/stage | Dropout | Epochs | Config |
| --- | --- | --- | --- | --- | --- |
| d0 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_d0.yaml |
| d1 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 50 | customcnn_d1_50ep.yaml |
| d2 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_d2_lr3e4.yaml |
| d3 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_d3_lr3e3.yaml |
| d4 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_d4_batch32.yaml |
| d5 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_d5_wd1e3.yaml |
| r1 | [32, 64, 128, 256] | [1, 2, 2, 2] | 0.3000 | 30 | customcnn_r1_residual.yaml |

## Quy tắc chọn

- Sàng lọc seed 42 cho cả 4 ứng viên.
- Chọn top-2 theo val Macro-F1, chạy thêm seed 123 và 2024.
- Chọn theo mean val Macro-F1; nếu chênh < 0.005, ưu tiên ít tham số/FLOPs hơn.
- Không dùng test ở bất kỳ bước nào.

## Kết quả

_Chưa có run completed. Chạy các config trong `configs/rq1/selection/`._
