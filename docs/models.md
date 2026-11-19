# Models

Registry: `src/models/factory.py`. Config chỉ cần `model.name`.

## Custom CNN

```python
from src.models import build_custom_cnn

model = build_custom_cnn(
    num_classes=5,
    channels=(32, 64, 128, 256),
    convs_per_stage=(1, 2, 2, 2),
    dropout=0.3,
    residual=False,
)
```

`residual=True` thêm skip connection cấp stage (1×1 conv khi đổi số kênh), tăng nhẹ số tham số. Đây là biến thể debug R1.

| Stage | Layers | Output |
|---|---|---|
| Stem | Conv 3×3 (3→32), BN, ReLU, MaxPool | 32×112×112 |
| Stage 1 | 2× [Conv 3×3 (32→64), BN, ReLU], MaxPool | 64×56×56 |
| Stage 2 | 2× [Conv 3×3 (64→128), BN, ReLU], MaxPool | 128×28×28 |
| Stage 3 | 2× [Conv 3×3 (128→256), BN, ReLU], MaxPool | 256×14×14 |
| Pool | AdaptiveAvgPool2d(1) | 256 |
| Head | Dropout(0.3) → Linear(256→5) | 5 logits |

- ~1,17M tham số, ~2,1 GMAC ở input 224.
- Không softmax; dùng `CrossEntropyLoss`.
- Khởi tạo Kaiming cho Conv, Xavier cho Linear.
- Tên layer (`stem`, `stage1..3`, `classifier`) hỗ trợ Grad-CAM sau này.
- Grad-CAM mặc định dùng stage cuối; có thể chọn layer bằng `--layer stages.3`.

Cấu hình này được chọn qua bước validation-only; xem `docs/model_selection.md`.

## MobileNetV3_small

```python
from src.models import build_mobilenet_v3_small

model = build_mobilenet_v3_small(num_classes=5, pretrained=True, dropout=0.2)
```

- Pretrained ImageNet (`MobileNet_V3_Small_Weights.IMAGENET1K_V1`).
- Thay classifier cuối bằng `Linear(1024→5)`.
- ~1,53M tham số, ~0,06 GMAC.

Transfer learning:

```python
from src.models import freeze_backbone, unfreeze_last_blocks, build_param_groups

freeze_backbone(model)            # 3 epoch đầu: chỉ train head
unfreeze_last_blocks(model, 2)    # mở 2 block cuối
groups = build_param_groups(model, base_lr=1e-3, weight_decay=1e-4)
```

- Head: lr `1e-3`.
- Backbone mở: lr `1e-4` (scale 0.1).
- Training engine tự tạo lại optimizer sau khi unfreeze.

## Factory

```python
from src.models import build_model, count_parameters

model = build_model("custom_cnn", num_classes=5)
print(count_parameters(model))
```

Hỗ trợ: `custom_cnn`, `mobilenet_v3_small`. Tên khác sẽ raise `ValueError`.

## Config mẫu

```yaml
model:
  name: custom_cnn
  num_classes: 5
  channels: [32, 64, 128, 256]
  convs_per_stage: [1, 2, 2, 2]
  dropout: 0.3
  residual: false
```

```yaml
model:
  name: mobilenet_v3_small
  num_classes: 5
  pretrained: true
  dropout: 0.2
  freeze_epochs: 3
  unfreeze_blocks: 2
```
