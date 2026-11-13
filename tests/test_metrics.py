"""Test metric và file predictions."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.training.metrics import classification_metrics, efficiency_metrics, write_predictions


def test_classification_metrics() -> None:
    metrics = classification_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        ["a", "b"],
        loss=0.7,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["loss"] == pytest.approx(0.7)
    assert metrics["per_class"]["a"] == {
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
        "support": 2,
    }
    assert metrics["per_class"]["b"]["recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["n_samples"] == 4


def test_metrics_are_json_safe() -> None:
    import json

    metrics = classification_metrics([0, 0, 1], [0, 1, 1], ["a", "b"])
    json.dumps(metrics)


def test_efficiency_metrics() -> None:
    metrics = efficiency_metrics(
        params=1000, train_time_sec=30.0, inference_time_sec=0.5, n_images=100
    )
    assert metrics["params"] == 1000
    assert metrics["inference_ms_per_image"] == pytest.approx(5.0)
    assert metrics["throughput_images_per_sec"] == pytest.approx(200.0)


def test_write_predictions(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "test.csv"
    write_predictions(
        path,
        paths=["realwaste/RealWaste/Glass/x.jpg"],
        datasets=["realwaste"],
        y_true=[1],
        y_pred=[1],
        probabilities=[[0.1, 0.9]],
        class_names=["a", "b"],
    )

    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["true_label"] == "b"
    assert rows[0]["pred_label"] == "b"
    assert rows[0]["correct"] == "1"
    assert rows[0]["prob_b"] == "0.9"


def test_write_predictions_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_predictions(
            tmp_path / "x.csv",
            paths=["a"],
            datasets=["d"],
            y_true=[0],
            y_pred=[0],
            probabilities=[],
            class_names=["a"],
        )
