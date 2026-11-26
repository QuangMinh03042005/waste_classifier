"""Thu activation từng layer và vẽ feature map cho Custom CNN."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

CUSTOM_CNN_CONV_LAYERS: tuple[str, ...] = (
    "stem",
    "stages.0.0",
    "stages.1.0",
    "stages.1.1",
    "stages.2.0",
    "stages.2.1",
    "stages.3.0",
    "stages.3.1",
)

CUSTOM_CNN_POOL_LAYERS: tuple[str, ...] = (
    "stages.0.1",
    "stages.1.2",
    "stages.2.2",
    "stages.3.2",
    "pool",
)

CUSTOM_CNN_LAYERS: tuple[str, ...] = (
    *CUSTOM_CNN_CONV_LAYERS,
    *CUSTOM_CNN_POOL_LAYERS,
    "classifier",
)


def resolve_layer(model: nn.Module, name: str) -> nn.Module:
    """Tìm module theo tên dạng `stages.1.0`."""
    target: Any = model
    for part in name.split("."):
        if part.isdigit():
            target = target[int(part)]
        else:
            if not hasattr(target, part):
                raise ValueError(f"Không tìm thấy layer: {name}")
            target = getattr(target, part)
    if not isinstance(target, nn.Module):
        raise TypeError(f"Layer không phải nn.Module: {name}")
    return target


def collect_activations(
    model: nn.Module,
    image: torch.Tensor,
    layers: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Forward một ảnh và trả activation của các layer được chọn."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"image phải có shape (1, 3, H, W), nhận {tuple(image.shape)}")

    names = list(layers or CUSTOM_CNN_LAYERS)
    modules = {name: resolve_layer(model, name) for name in names}
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                captured[layer_name] = output.detach().cpu()

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    missing = [name for name in names if name not in captured]
    if missing:
        raise RuntimeError(f"Không bắt được activation: {missing}")
    return {name: captured[name] for name in names}


def normalize_map(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Chuẩn hoá một map về [0, 1] để hiển thị."""
    values = np.asarray(array, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def top_channels(tensor: torch.Tensor, k: int = 6) -> list[int]:
    """Chọn k kênh có phương sai cao nhất."""
    single = tensor[0] if tensor.dim() == 4 else tensor
    if single.dim() != 3:
        raise ValueError(f"tensor phải có shape (C, H, W), nhận {tuple(single.shape)}")
    channels = int(single.shape[0])
    if channels <= k:
        return list(range(channels))
    variances = single.flatten(1).var(dim=1)
    return [int(index) for index in variances.argsort(descending=True)[:k]]


def plot_layer_maps(
    activations: Mapping[str, torch.Tensor],
    *,
    max_channels: int = 6,
    cmap: str = "viridis",
    title: str | None = None,
) -> Figure:
    """Vẽ lưới: mỗi hàng một layer, mỗi cột một kênh và map trung bình."""
    entries = [(name, tensor) for name, tensor in activations.items() if tensor.dim() == 4]
    if not entries:
        raise ValueError("Không có activation 4D để vẽ")

    columns = max_channels + 1
    fig, axes = plt.subplots(
        len(entries),
        columns,
        figsize=(columns * 1.05, len(entries) * 1.25),
        squeeze=False,
    )

    strips: list[tuple[int, np.ndarray, int]] = []

    for row, (name, tensor) in enumerate(entries):
        single = tensor[0]
        channels, height, width = (int(size) for size in single.shape)
        row_axes = axes[row]

        if height == 1 and width == 1:
            values = normalize_map(single.flatten().numpy())
            strips.append((row, values, channels))
            row_axes[0].set_xticks([])
            row_axes[0].set_yticks([])
            for spine in row_axes[0].spines.values():
                spine.set_visible(False)
            for axis in row_axes[1:]:
                axis.axis("off")
        else:
            selected = top_channels(tensor, max_channels)
            for column, channel in enumerate(selected):
                axis = row_axes[column]
                axis.imshow(normalize_map(single[channel].numpy()), cmap=cmap)
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(f"kênh {channel}", fontsize=7)
            for column in range(len(selected), max_channels):
                row_axes[column].axis("off")

            mean_axis = row_axes[-1]
            mean_axis.imshow(normalize_map(single.mean(dim=0).numpy()), cmap=cmap)
            mean_axis.set_xticks([])
            mean_axis.set_yticks([])
            if row == 0:
                mean_axis.set_title("trung bình", fontsize=7)

        row_axes[0].set_ylabel(
            f"{name}\n{channels}×{height}×{width}",
            fontsize=7.5,
            rotation=0,
            ha="right",
            va="center",
            labelpad=34,
        )

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    for row, values, channels in strips:
        start = axes[row, 1].get_position()
        end = axes[row, -1].get_position()
        strip = fig.add_axes([start.x0, start.y0, end.x1 - start.x0, start.height])
        strip.imshow(values[None, :], aspect="auto", cmap=cmap)
        strip.set_xticks([])
        strip.set_yticks([])
        strip.set_title(f"{channels} kênh · 1×1", fontsize=7)

    return fig


def plot_logits(
    logits: torch.Tensor,
    class_names: list[str],
    *,
    title: str | None = None,
) -> Figure:
    """Vẽ logits và xác suất softmax của 5 lớp."""
    values = logits.detach().cpu().flatten().float()
    probabilities = torch.softmax(values, dim=0).numpy()
    scores = values.numpy()

    fig, axis = plt.subplots(figsize=(6.5, 2.8))
    bars = axis.bar(class_names, scores, color="#93c5fd", edgecolor="#1d4ed8")
    axis.set_ylabel("logit")
    axis.set_ylim(min(0.0, float(scores.min())) - 0.5, float(scores.max()) + 1.2)
    for bar, probability in zip(bars, probabilities, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"p={probability:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    best = int(values.argmax())
    bars[best].set_color("#22c55e")
    if title:
        axis.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig
