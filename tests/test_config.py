"""Test config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.config import ConfigError, load_config


def make_config(tmp_path: Path, **overrides) -> Path:
    data = {
        "experiment": "rq1_test",
        "rq": "rq1",
        "seed": 42,
        "data": {"splits_version": "v1"},
        "model": {"name": "custom_cnn"},
        "training": {"epochs": 1},
    }
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    config = load_config(make_config(tmp_path))
    assert config["experiment"] == "rq1_test"


def test_missing_key(tmp_path: Path) -> None:
    path = make_config(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.pop("training")
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="thiếu trường"):
        load_config(path)


def test_invalid_epochs(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="epochs"):
        load_config(make_config(tmp_path, training={"epochs": 0}))
