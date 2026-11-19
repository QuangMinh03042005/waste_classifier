# External validation — GD

Mục đích: đánh giá ba arm của RQ1 trên một nguồn **chưa từng dùng để train**,
sau khi mô hình final đã được khóa.

- Dataset: Garbage Dataset (GD), Kaggle `sumn2u/garbage-classification-v2`.
- License: MIT.
- 5 lớp dùng: cardboard, glass, metal, paper, plastic.
- Không dùng external test để chọn model, epoch hay threshold.
- Kết quả external là mục “External validation” của RQ1, không thay đổi kết luận chính.

## Chuẩn bị dữ liệu

- Ảnh gốc sau mapping 5 lớp: 7.010.
- Loại ảnh trùng/near-duplicate với split v1: **2.200** ảnh.
- Loại trùng nội bộ GD: **4** ảnh.
- Giữ lại: **4.806** ảnh external test.

| Lớp | Số ảnh |
|---|---:|
| cardboard | 1.022 |
| glass | 1.343 |
| metal | 530 |
| paper | 765 |
| plastic | 1.146 |
| **Tổng** | **4.806** |

Lưu ý: tỷ lệ trùng với split v1 khá lớn (31,4%). Đây là lý do phải dedup trước khi
dùng GD làm external test; nếu không, kết quả sẽ bị leakage.

- CSV: `data/splits/external/gd_test.csv`.
- Summary: `data/splits/external/gd_summary.json`.
- Ảnh gốc: `data/external/gd/original/` (không commit).

## Giao thức

1. RQ1 final đã chạy xong và test in-domain đã được đánh giá một lần.
2. Với mỗi checkpoint `best.pt` của 3 arm × 5 seed, chạy đánh giá trên GD.
3. Không train lại, không đổi hyperparameter.
4. Tổng hợp mean ± std theo arm.
5. So sánh với in-domain test.

## Kết quả

Phần này sẽ được cập nhật sau khi RQ1 final hoàn tất:

| Model | In-domain Macro-F1 | GD Macro-F1 | Drop |
|---|---:|---:|---:|
| Custom CNN | — | — | — |
| MobileNetV3_small scratch | — | — | — |
| MobileNetV3_small pretrained | — | — | — |

## Cách diễn giải

- Pretrained giảm ít hơn Custom CNN → transfer learning giúp robustness.
- Cả ba giảm mạnh → domain shift lớn, củng cố động lực của RQ3.
- Scratch ≈ pretrained trên external → pretrained không giúp ngoài miền.
- Phân tích per-class F1 và Grad-CAM để tìm lớp/trường hợp yếu.

## File sinh tự động

- `docs/RQ1/external_validation.md` — báo cáo.
- `docs/RQ1/external_validation.csv` — từng seed.
- `docs/RQ1/external_validation.summary.csv` — mean ± std.

Script:

- `scripts/prepare_external_gd.py`
- `scripts/evaluate_external.py`
