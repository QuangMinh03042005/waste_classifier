"""Test sinh tracking Excel cho RQ3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from track_rq3 import collect_runs, plan_rows, summarize, write_workbook  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _make_run(root: Path, experiment: str, seed: int, correct_rate: float) -> Path:
    run_dir = root / "rq3" / f"{experiment}_s{seed}"
    (run_dir / "predictions").mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": f"{experiment}_s{seed}",
                "experiment": experiment,
                "seed": seed,
                "status": "completed",
                "started_at": "2026-11-21T00:00:00+00:00",
                "duration_sec": 200.0,
                "git": {"commit": "abcdef1234567890", "branch": "feature/rq3"},
                "device": "cuda",
                "model": {"params": 1_522_981},
                "dataset": {"train": 2163, "val": 826, "test": 823, "split_csv": "x/mix_tn_00.csv"},
                "plan": {"plan_id": "rq3_source_mix"},
            }
        ),
        encoding="utf-8",
    )
    labels = ["cardboard", "glass", "metal", "paper", "plastic"]
    predictions = []
    for index in range(20):
        dataset = "realwaste" if index % 2 == 0 else "trashnet"
        true_label = labels[index % len(labels)]
        pred_label = true_label if (index / 20) < correct_rate else labels[(index + 1) % 5]
        predictions.append(
            {
                "path": f"raw/{dataset}/{index}.jpg",
                "dataset": dataset,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": int(true_label == pred_label),
            }
        )
    pd.DataFrame(predictions).to_csv(run_dir / "predictions" / "test.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_val": {"epoch": 7, "macro_f1": 0.9},
                "test": {"macro_f1": 0.8, "accuracy": 0.81, "balanced_accuracy": 0.8},
                "efficiency": {"inference_ms_per_image": 5.5},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python_version": "3.12.16",
                "platform": "Windows",
                "git": {"commit": "abcdef1234567890", "branch": "feature/rq3", "dirty": True},
                "torch": {
                    "version": "2.15.0+cu132",
                    "cuda_version": "13.2",
                    "cudnn_version": 92600,
                    "device_name": "NVIDIA GeForce RTX 5070",
                    "device_capability": "12.0",
                    "vram_gb": 11.91,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_collect_runs_doc_artifact(tmp_path: Path) -> None:
    _make_run(tmp_path, "rq3_mix_tn00", 42, correct_rate=1.0)
    _make_run(tmp_path, "rq3_mix_tn00", 123, correct_rate=0.5)

    rows = collect_runs(tmp_path)
    assert len(rows) == 2
    first = rows[0]
    assert first["Nhóm"] == "Mix-ratio"
    assert first["Experiment"] == "rq3_mix_tn00"
    assert first["Train"] == 2163
    assert first["Val Macro-F1"] == pytest.approx(0.9)
    assert first["Test Macro-F1"] == pytest.approx(0.8)
    assert first["GPU"] == "NVIDIA GeForce RTX 5070"
    assert first["Test F1 RealWaste"] is not None
    assert first["Test F1 TrashNet"] is not None
    assert first["Commit"] == "abcdef12"


def test_summarize_mean_std(tmp_path: Path) -> None:
    _make_run(tmp_path, "rq3_mix_tn00", 42, correct_rate=1.0)
    _make_run(tmp_path, "rq3_mix_tn00", 123, correct_rate=0.0)
    rows = collect_runs(tmp_path)
    summary = summarize(rows)

    assert len(summary) == 1
    record = summary[0]
    assert record["n_seeds"] == 2
    assert record["Test Macro-F1 mean"] == pytest.approx(0.8)
    assert record["Test F1 RealWaste std"] == pytest.approx(
        pd.Series([row["Test F1 RealWaste"] for row in rows]).std(ddof=1)
    )


def test_plan_rows_doc_tu_plan_that(tmp_path: Path) -> None:
    rows = plan_rows([ROOT / "plans" / "rq3" / "source_mix.yaml"])
    assert len(rows) == 4
    roles = [row["Vai trò"] for row in rows]
    assert roles == ["baseline", "variant", "variant", "variant"]
    assert rows[0]["Experiment"] == "rq3_mix_tn00"
    assert rows[1]["Seeds"] == "42, 123, 2024"


def test_write_workbook_tao_du_sheet(tmp_path: Path) -> None:
    _make_run(tmp_path, "rq3_mix_tn00", 42, correct_rate=1.0)
    rows = collect_runs(tmp_path)
    output = tmp_path / "tracking.xlsx"

    path = write_workbook(
        rows,
        summarize(rows),
        plan_rows([ROOT / "plans" / "rq3" / "source_mix.yaml"]),
        output,
    )
    assert path == output and path.is_file()

    workbook = load_workbook(path)
    assert workbook.sheetnames == ["Runs", "Tổng hợp", "Kế hoạch", "Môi trường"]
    assert workbook["Runs"]["A1"].value == "ID"
    assert workbook["Runs"]["A2"].value == "rq3_mix_tn00_s42"
    assert workbook["Kế hoạch"]["B2"].value == "baseline"
    assert workbook["Môi trường"]["C2"].value == "NVIDIA GeForce RTX 5070"
