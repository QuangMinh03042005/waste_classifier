"""Test transform ảnh."""

from __future__ import annotations

import torch
from PIL import Image

from src.data.transforms import build_transforms


def make_image() -> Image.Image:
    return Image.new("RGB", (300, 260), color=(120, 140, 160))


def test_eval_transform_shape_and_determinism() -> None:
    transform = build_transforms(train=False, image_size=224)
    image = make_image()
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_train_transform_shape() -> None:
    transform = build_transforms(train=True, image_size=128)
    output = transform(make_image())
    assert output.shape == (3, 128, 128)


def test_train_without_augmentation_is_deterministic() -> None:
    transform = build_transforms(train=True, image_size=224, augmentation=False)
    image = make_image()
    assert torch.equal(transform(image), transform(image))


def test_invalid_image_size() -> None:
    try:
        build_transforms(train=True, image_size=0)
    except ValueError as exc:
        assert "image_size" in str(exc)
    else:
        raise AssertionError("phải báo lỗi image_size")
