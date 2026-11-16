# Chọn cấu hình MobileNetV3_small

Bước chọn cấu hình chỉ dùng **validation**, không đánh giá test (`evaluate_test: false`).

## Ứng viên

| ID | Channels | Conv/stage | Dropout | Epochs | Config |
| --- | --- | --- | --- | --- | --- |
| m1_full_ft |  |  | 0.2000 | 12 | m1_full_ft.yaml |
| m2_freeze3 |  |  | 0.2000 | 12 | m2_freeze3.yaml |
| m3_dropout03 |  |  | 0.3000 | 12 | m3_dropout03.yaml |
| m4_freeze5 |  |  | 0.2000 | 12 | m4_freeze5.yaml |

## Quy tắc chọn

- Sàng lọc seed 42 cho cả 4 ứng viên.
- Chọn top-2 theo val Macro-F1, chạy thêm seed 123 và 2024.
- Chọn theo mean val Macro-F1; nếu chênh < 0.005, ưu tiên ít tham số/FLOPs hơn.
- Không dùng test ở bất kỳ bước nào.

## Kết quả

_Chưa có run completed. Chạy các config trong `configs/rq1/selection/`._
