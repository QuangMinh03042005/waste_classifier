"""Model registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from src.models.custom_cnn import build_custom_cnn
from src.models.mobilenet import build_mobilenet_v3_small

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "custom_cnn": build_custom_cnn,
    "mobilenet_v3_small": build_mobilenet_v3_small,
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Dựng model theo tên trong registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {name!r} không có trong registry. Hỗ trợ: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Đếm số tham số."""
    parameters = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)
