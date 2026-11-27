"""Load model/checkpoint và calibration cho demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision import transforms

from src.data.labels import CANONICAL_CLASSES
from src.data.transforms import build_transforms
from src.training.checkpoint import load_best
from src.training.efficiency import build_model_from_config
from src.utils.paths import data_root as default_data_root
from src.utils.paths import project_root


@dataclass
class ModelBundle:
    """Model đã load cùng metadata cần cho suy luận."""

    key: str
    label: str
    run_dir: Path
    model: torch.nn.Module
    class_names: list[str]
    image_size: int
    transform: Any
    display: Any
    data_root: Path
    device: torch.device
    temperature: float = 1.0
    threshold: float = 0.6


def _resolve_data_root(config: dict[str, Any]) -> Path:
    configured = (config.get("data") or {}).get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else project_root() / path
    return default_data_root()


def load_bundle(
    key: str,
    label: str,
    run_dir: Path,
    *,
    device: torch.device | None = None,
) -> ModelBundle:
    """Load model từ run dir và trả bundle suy luận."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run.get("config") or {}
    class_names = list((config.get("data") or {}).get("class_names") or CANONICAL_CLASSES)
    image_size = int((config.get("data") or {}).get("image_size", 224))
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config)
    load_best(run_dir / "checkpoints" / "best.pt", model, map_location=target)
    model.to(target).eval()
    return ModelBundle(
        key=key,
        label=label,
        run_dir=run_dir,
        model=model,
        class_names=class_names,
        image_size=image_size,
        transform=build_transforms(train=False, image_size=image_size),
        display=transforms.Compose(
            [transforms.Resize(round(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
        ),
        data_root=_resolve_data_root(config),
        device=target,
    )


def load_calibration(path: Path) -> dict[str, dict[str, float]]:
    """Đọc file calibration; trả {} nếu chưa có."""
    path = Path(path)
    if not path.is_file():
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(key): {str(name): float(value) for name, value in values.items()}
        for key, values in content.items()
        if isinstance(values, dict)
    }


def apply_calibration(
    bundles: dict[str, ModelBundle], calibration: dict[str, dict[str, float]]
) -> None:
    """Gán temperature/threshold vào bundle theo key."""
    for key, bundle in bundles.items():
        values = calibration.get(key) or {}
        bundle.temperature = float(values.get("temperature", 1.0))
        bundle.threshold = float(values.get("threshold", 0.6))
