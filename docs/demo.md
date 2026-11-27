# Demo Streamlit

Upload một ảnh, chọn một trong ba model RQ1, xem dự đoán đã calibration,
Grad-CAM và feature map.

## Chạy

```bash
uv run streamlit run app/main.py
```

Mặc định dùng checkpoint `seed 42`:

| Model | Run |
|---|---|
| MobileNetV3_small pretrained (mặc định) | `rq1_mobilenetv3s_full_p20_s42` |
| Custom CNN | `rq1_customcnn_p20_s42` |
| MobileNetV3_small scratch | `rq1_mobilenetv3s_scratch_p20_s42` |

## Calibration

Fit trên **validation 826 ảnh**, không dùng test:

```bash
uv run python scripts/fit_calibration.py
```

Ghi `configs/demo/calibration.yaml` gồm temperature và threshold từng model.
App áp `softmax(logits / T)` và cảnh báo nếu confidence dưới threshold.

Kết quả hiện tại:

| Model | Temperature | Threshold |
|---|---:|---:|
| Custom CNN | 1,4341 | 0,79 |
| MobileNet pretrained | 3,0428 | 0,74 |
| MobileNet scratch | 4,5890 | 0,90 |

## Thành phần

- `src/inference/model_loader.py`: load run/checkpoint, calibration.
- `src/inference/predict.py`: dự đoán top-k, denormalize.
- `src/inference/calibration.py`: temperature scaling, threshold, ECE.
- `app/main.py`: UI.
- `scripts/fit_calibration.py`: fit calibration.

## Giới hạn

- Demo dùng một seed, không phải ensemble hay mean của 5 seed.
- Ảnh điện thoại là domain khác RealWaste/TrashNet; accuracy thực tế có thể
  thấp hơn test in-domain.
- Không dùng test để fit temperature/threshold.
- Grad-CAM và feature map chỉ mang tính định tính.
