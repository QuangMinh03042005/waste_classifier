# Data pipeline

Code: `src/data/dataset.py`, `src/data/transforms.py`. Split: `data/splits/v1/split.csv`.

## Cách dùng

```python
from pathlib import Path

from src.data.dataset import create_dataloaders, create_datasets

loaders = create_dataloaders(
    data_root=Path("data"),                 # trên Colab: /content/data
    split_csv=Path("data/splits/v1/split.csv"),
    image_size=224,
    batch_size=64,
    num_workers=2,
    augmentation=True,
    seed=42,
)

images, labels = next(iter(loaders["train"]))
# images: (B, 3, 224, 224), labels: (B,)
```

Trên Colab, `split.csv` lấy từ git; data root lấy từ `WASTE_DATA_ROOT`.

## Dataset

`WasteDataset`:

- Lọc đúng split (`train`/`val`/`test`) từ CSV.
- Đường dẫn trong CSV tương đối so với data root (`raw/...`).
- Trả `(image_tensor, label_index)`.
- `dataset.sample(i)` trả metadata (`path`, `dataset`, `label`, `split`, `group_id`) để ghi predictions.
- `dataset.class_counts`, `dataset.dataset_counts` phục vụ báo cáo/class weighting.
- Kiểm tra ảnh thiếu ngay khi khởi tạo; báo lỗi rõ ràng.
- Raise lỗi nếu CSV có nhãn lạ ngoài `class_names`.

Thứ tự lớp cố định:

```text
cardboard, glass, metal, paper, plastic
```

## Transform

| Giai đoạn | Pipeline |
|---|---|
| Train + augmentation | RandomResizedCrop(224, scale 0.7–1.0), HFlip, Rotation ±15°, ColorJitter, ToTensor, Normalize, RandomErasing p=0.1 |
| Train không augmentation | Resize(256), CenterCrop(224), ToTensor, Normalize |
| Val/Test | Resize(256), CenterCrop(224), ToTensor, Normalize |

- Normalize bằng mean/std ImageNet cho cả Custom CNN và MobileNetV3_small.
- `image_size` cấu hình được; mặc định 224.
- RandomErasing chỉ bật khi train + augmentation.

## DataLoader

`create_dataloaders` trả dict `{"train", "val", "test"}`:

- Train: `shuffle=True`, generator seed cố định.
- Val/Test: `shuffle=False`.
- `pin_memory` tự bật nếu có CUDA.
- `drop_last=False`.
- `persistent_workers` bật khi `num_workers > 0`.

## Lưu ý

- Không đọc test set trong lúc train; chỉ dùng `loaders["test"]` ở bước đánh giá cuối.
- `data_root` phải trỏ tới thư mục chứa `raw/` (local: `data`, Colab: `/content/data`).
- Nếu thay đổi split hoặc mapping, tạo version mới; không sửa v1.
