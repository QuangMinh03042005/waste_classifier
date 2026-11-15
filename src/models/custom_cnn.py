"""Custom CNN baseline cho phân loại 5 lớp."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Conv 3x3 + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CustomCNN(nn.Module):
    """CNN huấn luyện từ đầu.

    Mặc định: channels (32, 64, 128, 256), convs_per_stage (1, 2, 2, 2).
    Mỗi stage kết thúc bằng MaxPool2d(2), sau đó Global Average Pooling.
    """

    def __init__(
        self,
        num_classes: int = 5,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        convs_per_stage: tuple[int, ...] = (1, 2, 2, 2),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"channels phải có 4 phần tử, nhận {channels}")
        if len(convs_per_stage) != 4 or any(count < 1 for count in convs_per_stage):
            raise ValueError(
                f"convs_per_stage phải có 4 phần tử >= 1, nhận {convs_per_stage}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout phải trong [0, 1), nhận {dropout}")

        self.channels = tuple(channels)
        self.convs_per_stage = tuple(convs_per_stage)
        self.num_classes = num_classes

        self.stem = ConvBlock(3, channels[0])
        stages: list[nn.Sequential] = []
        in_channels = channels[0]
        for out_channels, conv_count in zip(channels, convs_per_stage, strict=True):
            layers: list[nn.Module] = []
            for _ in range(conv_count):
                layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_custom_cnn(
    num_classes: int = 5,
    channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
    convs_per_stage: tuple[int, ...] | list[int] = (1, 2, 2, 2),
    dropout: float = 0.3,
    **_: object,
) -> CustomCNN:
    """Factory cho Custom CNN; bỏ qua kwargs không liên quan."""
    return CustomCNN(
        num_classes=num_classes,
        channels=tuple(channels),
        convs_per_stage=tuple(convs_per_stage),
        dropout=dropout,
    )
