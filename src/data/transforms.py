"""Transform ảnh cho train/validation/test.

Train: RandomResizedCrop + HFlip + rotation + color jitter (+ RandomErasing).
Eval: Resize 256 + CenterCrop 224.
Chuẩn hóa bằng mean/std ImageNet cho cả hai mô hình.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224


def _resize_size(image_size: int) -> int:
    return round(image_size * 256 / 224)


def build_transforms(
    *,
    train: bool,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augmentation: bool = True,
    rotation_degrees: float = 15.0,
    color_jitter: float = 0.2,
    random_erasing_p: float = 0.1,
) -> transforms.Compose:
    """Tạo pipeline transform; augmentation chỉ áp dụng cho train."""
    if image_size <= 0:
        raise ValueError(f"image_size phải > 0, nhận {image_size}")

    operations: list = []
    if train and augmentation:
        operations.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                    hue=min(color_jitter / 4, 0.1),
                ),
            ]
        )
    else:
        operations.extend(
            [
                transforms.Resize(_resize_size(image_size)),
                transforms.CenterCrop(image_size),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if train and augmentation and random_erasing_p > 0:
        operations.append(transforms.RandomErasing(p=random_erasing_p))

    return transforms.Compose(operations)
