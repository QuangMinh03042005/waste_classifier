"""Test inference và calibration cho demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.inference.calibration import choose_threshold, fit_temperature
from src.inference.model_loader import load_calibration
from src.inference.predict import denormalize, top_k


def test_top_k_sorted() -> None:
    probabilities = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
    classes = ["a", "b", "c", "d", "e"]
    result = top_k(probabilities, classes, k=3)
    assert [name for name, _ in result] == ["b", "c", "a"]
    assert result[0][1] == 0.5


def test_temperature_scaling_reduces_nll() -> None:
    rng = np.random.default_rng(42)
    logits = np.stack([rng.normal(5.0, 0.5, 200), rng.normal(-5.0, 0.5, 200)], axis=1)
    labels = np.zeros(200, dtype=int)
    temperature = fit_temperature(logits, labels)
    assert 0.05 <= temperature <= 10.0

    def nll(values: np.ndarray) -> float:
        scaled = torch.softmax(torch.as_tensor(values) / temperature, dim=1).numpy()
        return float(-np.log(scaled[np.arange(len(labels)), labels] + 1e-12).mean())

    raw = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    raw_nll = float(-np.log(raw[np.arange(len(labels)), labels] + 1e-12).mean())
    assert nll(logits) <= raw_nll + 1e-6


def test_choose_threshold_perfect_confidence() -> None:
    probabilities = np.array([[0.99, 0.01], [0.01, 0.99], [0.98, 0.02], [0.02, 0.98]] * 20)
    labels = np.array([0, 1, 0, 1] * 20)
    threshold = choose_threshold(probabilities, labels, min_accepted=10)
    assert 0.5 <= threshold <= 0.99


def test_denormalize_shape() -> None:
    tensor = torch.zeros(1, 3, 224, 224)
    array = denormalize(tensor)
    assert array.shape == (224, 224, 3)
    assert 0.0 <= float(array.min()) <= 1.0
    assert 0.0 <= float(array.max()) <= 1.0


def test_load_calibration(tmp_path: Path) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text(
        "run_a:\n  temperature: 1.2\n  threshold: 0.7\n",
        encoding="utf-8",
    )
    calibration = load_calibration(path)
    assert calibration["run_a"]["temperature"] == 1.2
    assert calibration["run_a"]["threshold"] == 0.7
