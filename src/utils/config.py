"""Đọc và kiểm tra config YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("experiment", "rq", "seed", "data", "model", "training")


class ConfigError(ValueError):
    """Config thiếu hoặc sai định dạng."""


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: config phải là mapping")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ConfigError(f"{path}: thiếu trường {missing}")
    if not isinstance(data["data"], dict) or not isinstance(data["model"], dict):
        raise ConfigError(f"{path}: 'data' và 'model' phải là mapping")
    if not isinstance(data["training"], dict):
        raise ConfigError(f"{path}: 'training' phải là mapping")
    epochs = data["training"].get("epochs")
    if not isinstance(epochs, int) or epochs < 1:
        raise ConfigError(f"{path}: training.epochs phải là số nguyên >= 1")
    return data
