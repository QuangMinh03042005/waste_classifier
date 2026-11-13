"""Test tổng hợp kết quả thực nghiệm."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_results import collect, summarize, to_markdown, write_outputs


def write_run(
    root: Path,
    run_id: str,
    seed: int,
    *,
    status: str = "completed",
    test_f1: float = 0.7,
) -> None:
    run_dir = root / "rq1" / run_id
    run_dir.mkdir(parents=True)
    run = {
        "run_id": run_id,
        "rq": "rq1",
        "experiment": "rq1_demo",
        "seed": seed,
        "status": status,
        "started_at": "2026-11-13T00:00:00+00:00",
        "duration_sec": 100,
        "best_epoch": 3,
        "model": {"name": "mobile_net", "pretrained": "imagenet", "params": 1000},
        "config": {
            "augmentation": {"enabled": True},
            "class_weighting": False,
            "image_size": 224,
            "dataset": {"splits_version": "v1", "train": ["realwaste"]},
        },
        "git": {"commit": "abc", "branch": "main"},
    }
    metrics = {
        "best_val": {"epoch": 3, "macro_f1": 0.8, "loss": 0.5},
        "test": {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "precision_macro": 0.7,
            "recall_macro": 0.7,
            "macro_f1": test_f1,
            "per_class": {
                "glass": {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
            },
        },
        "efficiency": {
            "params": 1000,
            "train_time_sec": 10.0,
            "inference_ms_per_image": 5.0,
        },
    }
    environment = {
        "packages": {"torch": "2.15.0"},
        "torch": {"cuda_version": "13.2", "device_name": "Tesla T4"},
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment), encoding="utf-8")


def test_collect_and_summarize(tmp_path: Path) -> None:
    write_run(tmp_path, "rq1_demo_s42", 42, test_f1=0.7)
    write_run(tmp_path, "rq1_demo_s43", 43, test_f1=0.8)
    write_run(tmp_path, "rq1_failed_s42", 42, status="failed")

    results, per_class = collect(tmp_path)
    assert len(results) == 2
    assert set(results["run_id"]) == {"rq1_demo_s42", "rq1_demo_s43"}
    assert results.iloc[0]["model"] == "mobile_net"
    assert bool(results.iloc[0]["augmentation"]) is True
    assert results.iloc[0]["torch_version"] == "2.15.0"
    assert results.iloc[0]["gpu_name"] == "Tesla T4"

    assert len(per_class) == 2
    assert per_class.iloc[0]["class"] == "glass"

    summary = summarize(results)
    assert len(summary) == 1
    assert summary.iloc[0]["n_seeds"] == 2
    assert summary.iloc[0]["test_macro_f1_mean"] == 0.75
    assert summary.iloc[0]["test_macro_f1_std"] > 0


def test_include_failed(tmp_path: Path) -> None:
    write_run(tmp_path, "rq1_demo_s42", 42)
    write_run(tmp_path, "rq1_failed_s42", 42, status="failed")
    results, _ = collect(tmp_path, include_failed=True)
    assert len(results) == 2


def test_write_outputs(tmp_path: Path) -> None:
    write_run(tmp_path, "rq1_demo_s42", 42)
    results, per_class = collect(tmp_path)
    summary = summarize(results)
    out = tmp_path / "out"
    paths = write_outputs(out, results, per_class, summary, markdown=True)

    assert all(path.is_file() for path in paths.values())
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "# Kết quả thực nghiệm" in markdown
    assert "| experiment |" in markdown


def test_to_markdown_empty() -> None:
    import pandas as pd

    assert to_markdown(pd.DataFrame()) == "_Không có dữ liệu._"
