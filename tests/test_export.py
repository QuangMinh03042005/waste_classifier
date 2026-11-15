"""Test export experiments zip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.utils.export import create_export, select_runs


def make_run(
    root: Path,
    run_id: str,
    *,
    status: str = "completed",
    experiment: str = "rq1_demo",
) -> Path:
    run_dir = root / "rq1" / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    run = {
        "run_id": run_id,
        "rq": "rq1",
        "experiment": experiment,
        "seed": 42,
        "status": status,
        "started_at": "2026-11-16T00:00:00+00:00",
        "duration_sec": 100,
        "git": {"commit": "abc123"},
        "model": {"name": "custom_cnn", "params": 1000},
    }
    metrics = {
        "best_val": {"epoch": 3, "macro_f1": 0.8},
        "test": {"macro_f1": 0.7},
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.yaml").write_text("experiment: demo\n", encoding="utf-8")
    (run_dir / "history.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
    (run_dir / "predictions" / "test.csv").write_text("path\n", encoding="utf-8")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best")
    (run_dir / "checkpoints" / "last.pt").write_bytes(b"last")
    return run_dir


def test_select_runs_filters_status(tmp_path: Path) -> None:
    make_run(tmp_path, "run_ok_s42")
    make_run(tmp_path, "run_failed_s42", status="failed")

    assert [path.name for path in select_runs(tmp_path)] == ["run_ok_s42"]
    assert len(select_runs(tmp_path, include_incomplete=True)) == 2
    assert [path.name for path in select_runs(tmp_path, experiment="rq1_demo")] == [
        "run_ok_s42"
    ]
    assert select_runs(tmp_path, experiment="other") == []


def test_create_export_best_only(tmp_path: Path) -> None:
    make_run(tmp_path, "run_a_s42")
    make_run(tmp_path, "run_b_s42")

    output = tmp_path / "exports" / "runs.zip"
    info = create_export(tmp_path, output, select_runs(tmp_path))

    assert info.runs == 2
    assert info.path == output
    assert output.with_suffix(".zip.sha256").is_file()

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("experiments/_export/manifest.json"))

    assert "experiments/rq1/run_a_s42/config.yaml" in names
    assert "experiments/rq1/run_a_s42/checkpoints/best.pt" in names
    assert "experiments/rq1/run_a_s42/checkpoints/last.pt" not in names
    assert "experiments/_export/README.txt" in names
    assert manifest["totals"]["runs"] == 2
    assert manifest["checkpoints"] == "best"
    assert manifest["runs"][0]["run_id"] == "run_a_s42"


def test_create_export_all_checkpoints(tmp_path: Path) -> None:
    make_run(tmp_path, "run_a_s42")
    output = tmp_path / "runs_all.zip"
    create_export(
        tmp_path,
        output,
        select_runs(tmp_path),
        checkpoints="all",
    )
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "experiments/rq1/run_a_s42/checkpoints/best.pt" in names
    assert "experiments/rq1/run_a_s42/checkpoints/last.pt" in names


def test_create_export_none_checkpoints(tmp_path: Path) -> None:
    make_run(tmp_path, "run_a_s42")
    output = tmp_path / "runs_none.zip"
    create_export(tmp_path, output, select_runs(tmp_path), checkpoints="none")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert not any("/checkpoints/" in name for name in names)


def test_create_export_requires_force(tmp_path: Path) -> None:
    make_run(tmp_path, "run_a_s42")
    output = tmp_path / "runs.zip"
    create_export(tmp_path, output, select_runs(tmp_path))
    with pytest.raises(FileExistsError):
        create_export(tmp_path, output, select_runs(tmp_path))
    create_export(tmp_path, output, select_runs(tmp_path), force=True)
