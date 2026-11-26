# Layer summary: custom_cnn

Input: 1×3×224×224; sinh bởi `scripts/model_summary.py`.

| Layer | Type | Params | Output | MACs (M) |
|---|---|---:|---|---:|
| `stem.0` | Conv2d | 864 | 32×224×224 | 43.352 |
| `stem.1` | BatchNorm2d | 64 | 32×224×224 | 0.000 |
| `stem.2` | ReLU | 0 | 32×224×224 | 0.000 |
| `stages.0.0.0` | Conv2d | 9,216 | 32×224×224 | 462.422 |
| `stages.0.0.1` | BatchNorm2d | 64 | 32×224×224 | 0.000 |
| `stages.0.0.2` | ReLU | 0 | 32×224×224 | 0.000 |
| `stages.0.1` | MaxPool2d | 0 | 32×112×112 | 0.000 |
| `stages.1.0.0` | Conv2d | 18,432 | 64×112×112 | 231.211 |
| `stages.1.0.1` | BatchNorm2d | 128 | 64×112×112 | 0.000 |
| `stages.1.0.2` | ReLU | 0 | 64×112×112 | 0.000 |
| `stages.1.1.0` | Conv2d | 36,864 | 64×112×112 | 462.422 |
| `stages.1.1.1` | BatchNorm2d | 128 | 64×112×112 | 0.000 |
| `stages.1.1.2` | ReLU | 0 | 64×112×112 | 0.000 |
| `stages.1.2` | MaxPool2d | 0 | 64×56×56 | 0.000 |
| `stages.2.0.0` | Conv2d | 73,728 | 128×56×56 | 231.211 |
| `stages.2.0.1` | BatchNorm2d | 256 | 128×56×56 | 0.000 |
| `stages.2.0.2` | ReLU | 0 | 128×56×56 | 0.000 |
| `stages.2.1.0` | Conv2d | 147,456 | 128×56×56 | 462.422 |
| `stages.2.1.1` | BatchNorm2d | 256 | 128×56×56 | 0.000 |
| `stages.2.1.2` | ReLU | 0 | 128×56×56 | 0.000 |
| `stages.2.2` | MaxPool2d | 0 | 128×28×28 | 0.000 |
| `stages.3.0.0` | Conv2d | 294,912 | 256×28×28 | 231.211 |
| `stages.3.0.1` | BatchNorm2d | 512 | 256×28×28 | 0.000 |
| `stages.3.0.2` | ReLU | 0 | 256×28×28 | 0.000 |
| `stages.3.1.0` | Conv2d | 589,824 | 256×28×28 | 462.422 |
| `stages.3.1.1` | BatchNorm2d | 512 | 256×28×28 | 0.000 |
| `stages.3.1.2` | ReLU | 0 | 256×28×28 | 0.000 |
| `stages.3.2` | MaxPool2d | 0 | 256×14×14 | 0.000 |
| `pool` | AdaptiveAvgPool2d | 0 | 256×1×1 | 0.000 |
| `classifier.0` | Dropout | 0 | 256 | 0.000 |
| `classifier.1` | Linear | 1,285 | 5 | 0.001 |
| **Tổng** | | **1,174,501** | | **2586.67** |

## Tổng hợp theo stage

| Stage | Layers | Params | MACs (M) |
|---|---|---:|---:|
| `stem` | 3 | 928 | 43.352 |
| `stages.0` | 4 | 9,280 | 462.422 |
| `stages.1` | 7 | 55,552 | 693.633 |
| `stages.2` | 7 | 221,696 | 693.633 |
| `stages.3` | 7 | 885,760 | 693.633 |
| `pool` | 1 | 0 | 0.000 |
| `classifier` | 2 | 1,285 | 0.001 |
| **Tổng** | **31** | **1,174,501** | **2586.67** |

GFLOPs (2 × MACs): **5.173**.

