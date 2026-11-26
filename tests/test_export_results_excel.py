"""Test xuất Excel tổng hợp kết quả."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image

from scripts.export_results_excel import (
    _normalize,
    _safe_sheet_name,
    build_workbook,
)


def test_safe_sheet_name_unique() -> None:
    used: set[str] = set()
    first = _safe_sheet_name("rq1/experiment:test", used)
    second = _safe_sheet_name("rq1/experiment:test", used)
    assert first != second
    assert len(first) <= 31
    assert all(character not in first for character in "\\/*?:[]")


def test_normalize_rows() -> None:
    matrix = np.array([[2, 2], [0, 0]], dtype=float)
    normalized = _normalize(matrix)
    assert normalized[0].tolist() == [0.5, 0.5]
    assert normalized[1].tolist() == [0.0, 0.0]


def test_build_workbook(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (120, 80), color="white").save(image_path)
    results = pd.DataFrame(
        [
            {
                "run_id": "rq1_demo_s42",
                "rq": "rq1",
                "experiment": "rq1_demo",
                "seed": 42,
                "status": "completed",
                "model": "custom_cnn",
                "pretrained": False,
                "params": 1000,
                "augmentation": False,
                "class_weighting": False,
                "best_epoch": 10,
                "best_val_macro_f1": 0.8,
                "test_macro_f1": 0.7,
                "test_accuracy": 0.7,
                "test_balanced_accuracy": 0.7,
                "test_precision_macro": 0.7,
                "test_recall_macro": 0.7,
                "duration_sec": 100.0,
                "inference_ms_per_image": 1.0,
                "git_commit": "abc",
            }
        ]
    )
    output = tmp_path / "results.xlsx"
    build_workbook(
        results,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        experiment_images={"rq1_demo": image_path},
        run_images={"rq1_demo_s42": image_path},
        output=output,
    )

    assert output.is_file()
    workbook = load_workbook(output)
    for name in ("README", "Runs", "Summary", "Experiments", "Figures"):
        assert name in workbook.sheetnames
    assert "rq1_demo" not in workbook.sheetnames
    assert workbook["Runs"]["A2"].value == "rq1_demo_s42"
    assert len(workbook["Experiments"]._images) == 1
