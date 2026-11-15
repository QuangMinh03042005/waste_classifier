"""MobileNetV3_small pretrained ImageNet và tiện ích transfer learning."""

from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet_v3_small(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.2,
    **_: object,
) -> nn.Module:
    """MobileNetV3_small, thay lớp cuối cho num_classes."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    classifier = model.classifier
    classifier[2] = nn.Dropout(p=dropout)
    classifier[3] = nn.Linear(classifier[3].in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng toàn bộ features, chỉ train classifier."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, blocks: int = 2) -> None:
    """Mở các block cuối của features."""
    if blocks <= 0:
        return
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def build_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    backbone_lr_scale: float = 0.1,
) -> list[dict[str, Any]]:
    """Chia tham số trainable thành head và backbone với lr khác nhau."""
    backbone: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("features."):
            backbone.append(parameter)
        else:
            head.append(parameter)

    groups: list[dict[str, Any]] = []
    if backbone:
        groups.append(
            {
                "params": backbone,
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            }
        )
    if head:
        groups.append({"params": head, "lr": base_lr, "weight_decay": weight_decay})
    return groups
