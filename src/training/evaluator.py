"""Đánh giá một checkpoint trên một split/nguồn dữ liệu bất kỳ.

Dùng cho RQ3 (test ngoài miền) và các phân tích bổ sung, không cần train lại.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import WasteDataset
from src.data.labels import CANONICAL_CLASSES
from src.data.transforms import build_transforms
from src.models import build_model, count_parameters
from src.training.checkpoint import load_best
from src.training.metrics import classification_metrics, write_predictions
from src.utils.paths import data_root as default_data_root
from src.utils.paths import project_root, splits_dir


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    metrics_path: Path
    predictions_path: Path


def _resolve_data_root(config: dict[str, Any]) -> Path:
    configured = config["data"].get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (project_root() / path)
    return default_data_root()


def _resolve_split_csv(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    configured = config["data"].get("split_csv")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (project_root() / path)
    version = str(config["data"].get("splits_version", "v1"))
    return splits_dir() / version / "split.csv"


def _build_model_from_config(config: dict[str, Any], class_names: list[str]) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    name = str(model_cfg.pop("name"))
    num_classes = int(model_cfg.pop("num_classes", len(class_names)))
    model_cfg["pretrained"] = False  # chỉ dựng kiến trúc, weights lấy từ checkpoint
    return build_model(name, num_classes=num_classes, **model_cfg)


@torch.no_grad()
def evaluate_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    split: str,
    datasets: list[str] | None,
    output_dir: Path,
    split_csv: Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
) -> EvaluationResult:
    data_cfg = config["data"]
    class_names = list(data_cfg.get("class_names") or CANONICAL_CLASSES)
    image_size = int(data_cfg.get("image_size", 224))
    batch = int(batch_size or data_cfg.get("batch_size", 64))
    workers = int(num_workers if num_workers is not None else data_cfg.get("num_workers", 0))
    data_root = _resolve_data_root(config)
    csv_path = _resolve_split_csv(config, split_csv)

    dataset = WasteDataset(
        data_root=data_root,
        split_csv=csv_path,
        split=split,
        class_names=class_names,
        transform=build_transforms(train=False, image_size=image_size),
        datasets=datasets,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model_from_config(config, class_names)
    load_best(checkpoint, model, map_location=target_device)
    model.to(target_device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    use_amp = target_device.type == "cuda"

    for images, labels in loader:
        images = images.to(target_device, non_blocking=True)
        with torch.autocast(device_type=target_device.type, enabled=use_amp):
            logits = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(
        {
            "schema_version": "1.0",
            "checkpoint": str(checkpoint),
            "split": split,
            "datasets": datasets,
            "split_csv": str(csv_path),
            "params": count_parameters(model),
        }
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path = write_predictions(
        output_dir / "predictions.csv",
        paths=[sample.path for sample in dataset.samples],
        datasets=[sample.dataset for sample in dataset.samples],
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
    )
    return EvaluationResult(
        metrics=metrics,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
    )
