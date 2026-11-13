"""Test cấu hình đường dẫn và metadata môi trường."""

from __future__ import annotations

from pathlib import Path

from src.utils import paths
from src.utils.env import environment_info


def test_default_paths_point_to_project_root() -> None:
    root = paths.project_root()
    assert paths.data_root() == root / "data"
    assert paths.raw_root() == root / "data" / "raw"
    assert paths.splits_dir() == root / "data" / "splits"
    assert paths.experiments_root() == root / "experiments"


def test_env_overrides(monkeypatch, tmp_path: Path) -> None:
    data = tmp_path / "content" / "data"
    experiments = tmp_path / "drive" / "experiments"
    monkeypatch.setenv("WASTE_DATA_ROOT", str(data))
    monkeypatch.setenv("WASTE_EXPERIMENTS_ROOT", str(experiments))
    monkeypatch.setenv("WASTE_SPLITS_DIR", str(tmp_path / "splits"))

    assert paths.data_root() == data.resolve()
    assert paths.raw_root() == (data / "raw").resolve()
    assert paths.dataset_dir("realwaste") == (data / "raw" / "realwaste").resolve()
    assert paths.experiments_root() == experiments.resolve()
    assert paths.splits_dir() == (tmp_path / "splits").resolve()
    assert paths.experiment_dir("rq1", "run_s42") == (experiments / "rq1" / "run_s42").resolve()


def test_environment_info_has_required_fields() -> None:
    info = environment_info()
    assert info["python_version"]
    assert "git" in info
    assert "packages" in info
    assert info["torch"]["installed"] is True
