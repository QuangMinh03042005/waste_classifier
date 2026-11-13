"""Test logger của một run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.run_logger import RunExistsError, RunLogger, read_history


def make_logger(tmp_path: Path, **kwargs) -> RunLogger:
    return RunLogger(
        "rq1",
        "rq1_demo_s42",
        {"seed": 42, "experiment": "rq1_demo", "model": {"name": "demo"}},
        experiments_root=tmp_path,
        **kwargs,
    )


def test_create_run_writes_schema(tmp_path: Path) -> None:
    make_logger(tmp_path)
    root = tmp_path / "rq1" / "rq1_demo_s42"

    assert (root / "config.yaml").is_file()
    assert (root / "environment.json").is_file()
    assert (root / "checkpoints").is_dir()
    assert (root / "predictions").is_dir()

    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert run["schema_version"] == "1.0"
    assert run["status"] == "running"
    assert run["seed"] == 42
    assert run["experiment"] == "rq1_demo"
    assert run["git"]["commit"]


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    make_logger(tmp_path)
    with pytest.raises(RunExistsError):
        make_logger(tmp_path)


def test_log_epoch_and_history(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    logger.log_epoch(1, "train", {"loss": 1.0, "macro_f1": 0.5}, duration_sec=12.5)
    logger.log_epoch(
        1, "val", {"loss": 0.9, "macro_f1": 0.6, "per_class_f1": {"a": 0.5}}
    )

    history = read_history(logger.history)
    assert len(history) == 2
    assert history[0]["phase"] == "train"
    assert history[0]["duration_sec"] == 12.5
    assert history[1]["per_class_f1"] == {"a": 0.5}
    assert "at" in history[0]


def test_finish_and_fail(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    logger.finish(
        {
            "class_names": ["a", "b"],
            "best_val": {"epoch": 2, "macro_f1": 0.6, "loss": 0.9},
            "test": {"macro_f1": 0.55},
            "efficiency": {},
        }
    )

    run = json.loads(logger.run_json.read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["best_epoch"] == 2
    assert run["result"]["test_macro_f1"] == 0.55
    assert logger.metrics.is_file()

    failed = RunLogger(
        "rq1",
        "rq1_demo_s43",
        {"seed": 43, "experiment": "rq1_demo"},
        experiments_root=tmp_path,
    )
    failed.fail(ValueError("boom"))
    error = json.loads(failed.error.read_text(encoding="utf-8"))
    assert error["type"] == "ValueError"
    assert "boom" in error["message"]
    run = json.loads(failed.run_json.read_text(encoding="utf-8"))
    assert run["status"] == "failed"


def test_resume_updates_count(tmp_path: Path) -> None:
    make_logger(tmp_path)
    resumed = make_logger(tmp_path, resume=True, from_epoch=3)
    run = json.loads(resumed.run_json.read_text(encoding="utf-8"))
    assert run["status"] == "running"
    assert run["resume"] == {"count": 1, "from_epoch": 3}


def test_invalid_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        RunLogger(
            "rq1",
            "../escape",
            {"seed": 1},
            experiments_root=tmp_path,
        )
