"""Test thu activation và vẽ feature map."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
import torch

from src.models import build_custom_cnn
from src.training.feature_maps import (
    CUSTOM_CNN_LAYERS,
    collect_activations,
    normalize_map,
    plot_layer_maps,
    plot_logits,
    resolve_layer,
    top_channels,
)


@pytest.fixture(scope="module")
def model():
    return build_custom_cnn().eval()


def test_activation_shapes(model) -> None:
    activations = collect_activations(model, torch.zeros(1, 3, 224, 224))
    shapes = {name: tuple(tensor.shape) for name, tensor in activations.items()}

    assert shapes["stem"] == (1, 32, 224, 224)
    assert shapes["stages.0.0"] == (1, 32, 224, 224)
    assert shapes["stages.0.1"] == (1, 32, 112, 112)
    assert shapes["stages.1.0"] == (1, 64, 112, 112)
    assert shapes["stages.1.1"] == (1, 64, 112, 112)
    assert shapes["stages.1.2"] == (1, 64, 56, 56)
    assert shapes["stages.2.0"] == (1, 128, 56, 56)
    assert shapes["stages.2.1"] == (1, 128, 56, 56)
    assert shapes["stages.2.2"] == (1, 128, 28, 28)
    assert shapes["stages.3.0"] == (1, 256, 28, 28)
    assert shapes["stages.3.1"] == (1, 256, 28, 28)
    assert shapes["stages.3.2"] == (1, 256, 14, 14)
    assert shapes["pool"] == (1, 256, 1, 1)
    assert shapes["classifier"] == (1, 5)
    assert list(shapes) == list(CUSTOM_CNN_LAYERS)


def test_collect_restores_training_mode(model) -> None:
    model.train()
    collect_activations(model, torch.zeros(1, 3, 224, 224))
    assert model.training is True


def test_collect_rejects_batch(model) -> None:
    with pytest.raises(ValueError, match="shape"):
        collect_activations(model, torch.zeros(2, 3, 224, 224))


def test_resolve_layer_invalid(model) -> None:
    with pytest.raises(ValueError, match="Không tìm thấy"):
        resolve_layer(model, "khong_ton_tai")
    with pytest.raises(TypeError, match="nn.Module"):
        resolve_layer(model, "channels")


def test_normalize_map() -> None:
    constant = normalize_map(torch.ones(4, 4))
    assert constant.shape == (4, 4)
    assert float(constant.max()) == 0.0

    varied = normalize_map(torch.arange(9, dtype=torch.float32).reshape(3, 3))
    assert float(varied.min()) == 0.0
    assert float(varied.max()) == 1.0


def test_top_channels() -> None:
    tensor = torch.randn(1, 32, 8, 8)
    selected = top_channels(tensor, k=6)
    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert all(0 <= index < 32 for index in selected)

    small = top_channels(torch.randn(1, 3, 4, 4), k=6)
    assert small == [0, 1, 2]


def test_plot_layer_maps(model) -> None:
    activations = collect_activations(
        model,
        torch.zeros(1, 3, 64, 64),
        layers=["stem", "stages.0.0", "pool"],
    )
    figure = plot_layer_maps(activations, max_channels=4, title="test")
    # 3 hàng × 5 cột, cộng thêm 1 axes dải cho layer GAP 1×1
    assert len(figure.axes) == 3 * (4 + 1) + 1
    plt.close(figure)


def test_plot_logits() -> None:
    figure = plot_logits(
        torch.tensor([[0.1, 0.5, 2.0, -0.3, 0.0]]),
        ["cardboard", "glass", "metal", "paper", "plastic"],
    )
    assert len(figure.axes) == 1
    plt.close(figure)
