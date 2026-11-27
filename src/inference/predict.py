"""Dự đoán top-k cho demo."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.inference.model_loader import ModelBundle


def predict_proba(
    bundle: ModelBundle,
    image: Image.Image,
    *,
    apply_temperature: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """Trả xác suất và logits của một ảnh."""
    tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(tensor)
        scaled = logits / bundle.temperature if apply_temperature else logits
        probabilities = F.softmax(scaled, dim=1)
    return probabilities[0].detach().cpu().numpy(), logits[0].detach().cpu()


def top_k(probabilities: np.ndarray, class_names: list[str], k: int = 3) -> list[tuple[str, float]]:
    """Top-k lớp theo xác suất giảm dần."""
    order = np.argsort(-probabilities)[:k]
    return [(class_names[int(index)], float(probabilities[int(index)])) for index in order]


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Đưa tensor ImageNet về ảnh RGB [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor[0] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def model_info(bundle: ModelBundle) -> dict[str, Any]:
    """Thông tin ngắn để hiển thị."""
    params = sum(parameter.numel() for parameter in bundle.model.parameters())
    return {
        "params": params,
        "temperature": bundle.temperature,
        "threshold": bundle.threshold,
        "device": str(bundle.device),
    }
