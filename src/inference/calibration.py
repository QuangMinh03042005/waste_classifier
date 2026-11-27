"""Temperature scaling và chọn threshold trên validation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit T > 0 để giảm NLL; trả T trong [0.05, 10]."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(0.05, 10.0).item())


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    target_precision: float = 0.95,
    min_accepted: int = 50,
    fallback: float = 0.6,
) -> float:
    """Chọn threshold nhỏ nhất đạt precision mục tiêu trên validation."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(float)
    best = fallback
    for threshold in np.linspace(0.5, 0.99, 50):
        accepted = confidence >= threshold
        if accepted.sum() < min_accepted:
            continue
        precision = float(correct[accepted].mean())
        if precision >= target_precision:
            best = float(threshold)
            break
    return best


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """ECE cho báo cáo nhanh trên validation."""
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracy = (predictions == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(accuracy[mask].mean()) - float(confidence[mask].mean())
        )
    return error
