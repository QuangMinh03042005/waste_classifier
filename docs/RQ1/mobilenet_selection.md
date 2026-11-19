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
| m1_full_ft_50 | 123 | 0.8974 | 0.5293 | 17 | 1522981 | 681.0540 | 791ed59e |
| m1_full_ft_50 | 2024 | 0.9005 | 0.5523 | 29 | 1522981 | 1009.6570 | 791ed59e |
| m1_full_ft_50 | 42 | 0.8809 | 0.4367 | 7 | 1522981 | 397.9550 | 791ed59e |
| m1_full_ft | 42 | 0.8811 | 0.4107 | 9 | 1522981 | 344.7210 | c01de168 |
| m2_freeze3 | 42 | 0.8505 | 0.4374 | 10 | 1522981 | 325.0340 | c01de168 |
| m3_dropout03 | 42 | 0.8421 | 0.4423 | 10 | 1522981 | 321.8630 | c01de168 |
| m4_freeze5_50 | 123 | 0.8732 | 0.5300 | 21 | 1522981 | 769.5350 | 791ed59e |
| m4_freeze5_50 | 2024 | 0.8726 | 0.5487 | 26 | 1522981 | 881.2700 | 791ed59e |
| m4_freeze5_50 | 42 | 0.8830 | 0.5032 | 24 | 1522981 | 852.5240 | 791ed59e |
| m4_freeze5 | 42 | 0.8577 | 0.4294 | 10 | 1522981 | 322.5680 | c01de168 |

## Tổng hợp theo ứng viên

| ID | Số seed | Val Macro-F1 mean | Std | Params |
| --- | --- | --- | --- | --- |
| m1_full_ft | 1 | 0.8811 |  | 1522981 |
| m1_full_ft_50 | 3 | 0.8929 | 0.0105 | 1522981 |
| m2_freeze3 | 1 | 0.8505 |  | 1522981 |
| m3_dropout03 | 1 | 0.8421 |  | 1522981 |
| m4_freeze5 | 1 | 0.8577 |  | 1522981 |
| m4_freeze5_50 | 3 | 0.8762 | 0.0059 | 1522981 |

## Kết luận

- Chọn **m1_full_ft_50** với val Macro-F1 trung bình **0.8929** (std 0.0105), 1522981 tham số.
- Đây là kết quả chọn trên validation; test chưa được đánh giá.
- Dùng cấu hình này cho RQ1 với budget cuối × 5 seed ở bước tiếp theo.
