# Kết quả RQ1

Sinh tự động bởi `scripts/analyze_rq1.py`; test chỉ được đánh giá một lần.

## Tổng hợp

| label | n_seeds | Macro-F1 mean | Macro-F1 std | Accuracy mean | Params |
| --- | --- | --- | --- | --- | --- |
| MobileNetV3_small pretrained | 5 | 0.8901 | 0.0114 | 0.8914 | 1522981.0000 |
| Custom CNN | 5 | 0.8870 | 0.0060 | 0.8868 | 1174501.0000 |

## Per-class F1

| label | class | f1_mean | f1_std | recall_mean |
| --- | --- | --- | --- | --- |
| Custom CNN | cardboard | 0.8862 | 0.0133 | 0.8738 |
| Custom CNN | glass | 0.8817 | 0.0112 | 0.9014 |
| Custom CNN | metal | 0.8661 | 0.0066 | 0.8522 |
| Custom CNN | paper | 0.9138 | 0.0138 | 0.9049 |
| Custom CNN | plastic | 0.8870 | 0.0129 | 0.9005 |
| MobileNetV3_small pretrained | cardboard | 0.8836 | 0.0214 | 0.8754 |
| MobileNetV3_small pretrained | glass | 0.8786 | 0.0067 | 0.8797 |
| MobileNetV3_small pretrained | metal | 0.8733 | 0.0134 | 0.8500 |
| MobileNetV3_small pretrained | paper | 0.9124 | 0.0133 | 0.9585 |
| MobileNetV3_small pretrained | plastic | 0.9024 | 0.0131 | 0.8919 |

## Lỗi phổ biến

| label | true_label | pred_label | count |
| --- | --- | --- | --- |
| Custom CNN | metal | plastic | 60.0000 |
| Custom CNN | plastic | metal | 44.0000 |
| Custom CNN | metal | glass | 40.0000 |
| Custom CNN | plastic | glass | 37.0000 |
| Custom CNN | cardboard | paper | 36.0000 |
| Custom CNN | [realwaste] | error_rate | 0.1011 |
| Custom CNN | [trashnet] | error_rate | 0.1289 |
| MobileNetV3_small pretrained | cardboard | paper | 64.0000 |
| MobileNetV3_small pretrained | glass | plastic | 47.0000 |
| MobileNetV3_small pretrained | plastic | metal | 47.0000 |
| MobileNetV3_small pretrained | metal | glass | 43.0000 |
| MobileNetV3_small pretrained | plastic | glass | 40.0000 |
| MobileNetV3_small pretrained | [realwaste] | error_rate | 0.1050 |
| MobileNetV3_small pretrained | [trashnet] | error_rate | 0.1133 |

## Ghi chú trạng thái

- Scratch arm cũ **không hợp lệ**: backbone bị scale LR như fine-tuning, nhiều seed
  gần mức random. Các run này bị loại khỏi phân tích.
- Scratch đang được sửa recipe và sẽ cập nhật sau; không dùng kết quả scratch cũ.
- Bảng trên chỉ gồm Custom CNN và MobileNet pretrained.

## Kết luận RQ1 (tạm thời)

- Custom CNN: Macro-F1 **0.8870 ± 0.0060**.
- MobileNet pretrained: **0.8901 ± 0.0114**.
- Chênh lệch trung bình: **+0.0031**.
- Pretrained thắng **3/5 seed**.
- Không đạt tiêu chí **≥ 0.02** và **≥ 4/5 seed**.
- Giả thuyết “pretrained cải thiện rõ rệt” **không được ủng hộ** trên test.
- Tuy nhiên MobileNet chỉ dùng **0,11 GFLOPs** so với **5,17 GFLOPs** của Custom CNN.
  Nếu hiệu năng tương đương, MobileNet có lợi thế lớn về compute.

## Efficiency

- Custom CNN: ~5,17 GFLOPs, latency CPU ~28–37 ms/ảnh.
- MobileNetV3\_small: ~0,11 GFLOPs, latency CPU ~9–11 ms/ảnh.
- Chi tiết: `docs/RQ1/rq1_efficiency.md`.

## External GD

- Custom CNN: 0.3694 ± 0.0117.
- MobileNet pretrained: 0.5409 ± 0.0097.
- Chi tiết: `docs/RQ1/external_validation.md`.
- Pretrained giảm ít hơn khi đổi domain, dù cả hai đều giảm mạnh.

## Hình bổ sung

- `gradcam_custom_cnn.png`
- `gradcam_mobilenet.png`

## Scratch correction

- Plan: `plans/rq1/scratch_fix.yaml`.
- Config S1/S2: `configs/rq1/scratch_fix/`.
- Kết quả sẽ được cập nhật sau khi chạy trên GPU.
- Test reuse sẽ được công bố trong báo cáo.


