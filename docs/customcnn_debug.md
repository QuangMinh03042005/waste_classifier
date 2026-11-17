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

| ID | Seed | Val Macro-F1 | Val Loss | Best epoch | Params | Duration (s) | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d0 | 42 | 0.8579 | 0.4474 | 30 | 1174501 | 834.5820 | c01de168 |
| d1 | 42 | 0.8746 | 0.4121 | 48 | 1174501 | 1417.2560 | c01de168 |
| d2 | 42 | 0.8616 | 0.4852 | 21 | 1174501 | 781.2080 | c01de168 |
| d3 | 42 | 0.8413 | 0.4819 | 30 | 1174501 | 897.4890 | c01de168 |
| d4 | 42 | 0.8590 | 0.4494 | 29 | 1174501 | 887.2520 | c01de168 |
| d5 | 42 | 0.8622 | 0.4519 | 28 | 1174501 | 899.9390 | c01de168 |
| r1 | 42 | 0.8595 | 0.4272 | 30 | 1218405 | 923.1010 | c01de168 |

## Tổng hợp theo ứng viên

| ID | Số seed | Val Macro-F1 mean | Std | Params |
| --- | --- | --- | --- | --- |
| d0 | 1 | 0.8579 |  | 1174501 |
| d1 | 1 | 0.8746 |  | 1174501 |
| d2 | 1 | 0.8616 |  | 1174501 |
| d3 | 1 | 0.8413 |  | 1174501 |
| d4 | 1 | 0.8590 |  | 1174501 |
| d5 | 1 | 0.8622 |  | 1174501 |
| r1 | 1 | 0.8595 |  | 1218405 |

## Kết luận

- Chọn **d1** với val Macro-F1 trung bình **0.8746** (std ), 1174501 tham số.
- Đây là kết quả chọn trên validation; test chưa được đánh giá.
- Dùng cấu hình này cho RQ1 với budget cuối × 5 seed ở bước tiếp theo.
- Lưu ý: ứng viên này mới có ít hơn 3 seed.
