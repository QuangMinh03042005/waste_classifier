# Chọn cấu hình MobileNetV3_small

Bước chọn cấu hình chỉ dùng **validation**, không đánh giá test (`evaluate_test: false`).

## Ứng viên

| ID | Channels | Conv/stage | Dropout | Epochs | Config |
| --- | --- | --- | --- | --- | --- |
| m1_full_ft |  |  | 0.2000 | 12 | m1_full_ft.yaml |
| m1_full_ft_50 |  |  | 0.2000 | 50 | m1_full_ft_50ep.yaml |
| m2_freeze3 |  |  | 0.2000 | 12 | m2_freeze3.yaml |
| m3_dropout03 |  |  | 0.3000 | 12 | m3_dropout03.yaml |
| m4_freeze5 |  |  | 0.2000 | 12 | m4_freeze5.yaml |
| m4_freeze5_50 |  |  | 0.2000 | 50 | m4_freeze5_50ep.yaml |

## Quy tắc chọn

- Sàng lọc seed 42 cho cả 4 ứng viên.
- Chọn top-2 theo val Macro-F1, chạy thêm seed 123 và 2024.
- Chọn theo mean val Macro-F1; nếu chênh < 0.005, ưu tiên ít tham số/FLOPs hơn.
- Không dùng test ở bất kỳ bước nào.

## Kết quả

| ID | Seed | Val Macro-F1 | Val Loss | Best epoch | Params | Duration (s) | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m1_full_ft | 42 | 0.8811 | 0.4107 | 9 | 1522981 | 344.7210 | c01de168 |
| m2_freeze3 | 42 | 0.8505 | 0.4374 | 10 | 1522981 | 325.0340 | c01de168 |
| m3_dropout03 | 42 | 0.8421 | 0.4423 | 10 | 1522981 | 321.8630 | c01de168 |
| m4_freeze5 | 42 | 0.8577 | 0.4294 | 10 | 1522981 | 322.5680 | c01de168 |

## Tổng hợp theo ứng viên

| ID | Số seed | Val Macro-F1 mean | Std | Params |
| --- | --- | --- | --- | --- |
| m1_full_ft | 1 | 0.8811 |  | 1522981 |
| m2_freeze3 | 1 | 0.8505 |  | 1522981 |
| m3_dropout03 | 1 | 0.8421 |  | 1522981 |
| m4_freeze5 | 1 | 0.8577 |  | 1522981 |

## Kết luận

- Chọn **m1_full_ft** với val Macro-F1 trung bình **0.8811** (std ), 1522981 tham số.
- Đây là kết quả chọn trên validation; test chưa được đánh giá.
- Dùng cấu hình này cho RQ1 với budget cuối × 5 seed ở bước tiếp theo.
- Lưu ý: ứng viên này mới có ít hơn 3 seed.
