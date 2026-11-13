"""Tính metric phân loại và ghi file dự đoán."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    *,
    loss: float | None = None,
) -> dict:
    """Metric phân loại đa lớp, trả về dict JSON-serializable."""
    true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError(f"y_true và y_pred lệch kích thước: {true.shape} vs {pred.shape}")

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=labels, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true, pred, labels=labels, average="macro", zero_division=0
    )

    matrix = confusion_matrix(true, pred, labels=labels)
    row_totals = matrix.sum(axis=1)
    recalls = [
        float(matrix[i, i] / row_totals[i]) if row_totals[i] else 0.0
        for i in range(len(class_names))
    ]

    return {
        "loss": None if loss is None else float(loss),
        "accuracy": float(np.mean(true == pred)) if true.size else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(true.size),
    }


def efficiency_metrics(
    *,
    params: int,
    train_time_sec: float,
    inference_time_sec: float,
    n_images: int,
) -> dict:
    """Số tham số và thời gian suy luận."""
    latency = inference_time_sec / n_images if n_images else None
    return {
        "params": int(params),
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": float(inference_time_sec),
        "inference_ms_per_image": None if latency is None else latency * 1000,
        "throughput_images_per_sec": None if not latency else 1.0 / latency,
    }


def write_predictions(
    path: Path,
    *,
    paths: Sequence[str],
    datasets: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: Sequence[str],
) -> Path:
    """Ghi predictions CSV: nhãn thật, nhãn dự đoán, xác suất từng lớp."""
    if not (len(paths) == len(datasets) == len(y_true) == len(y_pred) == len(probabilities)):
        raise ValueError("Số lượng path/dataset/nhãn/xác suất không khớp nhau.")

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "path",
        "dataset",
        "true_label",
        "pred_label",
        "correct",
        *(f"prob_{name}" for name in class_names),
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row, dataset, label, prediction, probs in zip(
            paths, datasets, y_true, y_pred, probabilities, strict=True
        ):
            record = {
                "path": row,
                "dataset": dataset,
                "true_label": class_names[int(label)],
                "pred_label": class_names[int(prediction)],
                "correct": int(label == prediction),
            }
            for name, value in zip(class_names, probs, strict=True):
                record[f"prob_{name}"] = round(float(value), 6)
            writer.writerow(record)
    return path
