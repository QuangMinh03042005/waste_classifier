"""Grad-CAM cho Custom CNN và MobileNetV3."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


def resolve_target_layer(model: nn.Module, layer: str | None = None) -> nn.Module:
    """Chọn layer cuối của backbone mặc định."""
    if layer:
        target: Any = model
        for part in layer.split("."):
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        if not isinstance(target, nn.Module):
            raise TypeError(f"Layer không phải nn.Module: {layer}")
        return target

    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "stages"):
        return model.stages[-1]
    raise ValueError("Không xác định được layer cho Grad-CAM; truyền --layer.")


class GradCAM:
    """Tính Grad-CAM từ activation và gradient của layer mục tiêu."""

    def __init__(self, model: nn.Module, layer: str | None = None) -> None:
        self.model = model
        self.layer = resolve_target_layer(model, layer)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward = self.layer.register_forward_hook(self._capture_forward)
        self._backward = self.layer.register_full_backward_hook(self._capture_backward)

    def _capture_forward(self, module, inputs, output) -> None:
        self.activations = output

    def _capture_backward(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def generate(
        self, image: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if target_class is None:
            target_class = int(logits.argmax(dim=1)[0])
        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Không bắt được activation/gradient; kiểm tra layer.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        array = cam[0, 0].detach().cpu().numpy()
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        return array, int(target_class)

    def remove(self) -> None:
        self._forward.remove()
        self._backward.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    """Ghép heatmap lên ảnh, trả về RGB float [0,1]."""
    import matplotlib

    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    pixels = np.asarray(base, dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * pixels + alpha * heat, 0.0, 1.0)
