"""Test efficiency và Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.models import build_custom_cnn
from src.training.efficiency import measure_model
from src.training.gradcam import GradCAM, overlay_cam


def tiny_model():
    return build_custom_cnn(
        num_classes=5,
        channels=[4, 8, 16, 32],
        convs_per_stage=[1, 1, 1, 1],
        dropout=0.1,
    )


def test_measure_model() -> None:
    result = measure_model(tiny_model(), device=torch.device("cpu"), run_id="test")
    assert result.params > 0
    assert result.flops > 0
    assert result.gflops > 0
    assert result.latency_ms_batch1 > 0
    assert result.vram_mb is None


def test_gradcam() -> None:
    model = tiny_model()
    cam = GradCAM(model)
    image = torch.randn(1, 3, 32, 32)
    heatmap, target = cam.generate(image)
    cam.remove()

    assert heatmap.shape == (32, 32)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
    assert 0 <= target < 5


def test_overlay_cam() -> None:
    image = Image.new("RGB", (32, 32), color=(100, 120, 140))
    heatmap = np.zeros((32, 32), dtype=np.float32)
    overlay = overlay_cam(image, heatmap)
    assert overlay.shape == (32, 32, 3)
    assert 0.0 <= overlay.min() <= overlay.max() <= 1.0
