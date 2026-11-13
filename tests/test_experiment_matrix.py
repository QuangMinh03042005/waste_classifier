"""Test sinh ma trận thí nghiệm."""

from __future__ import annotations

from pathlib import Path

from scripts.experiment_matrix import build_markdown, write_matrix
from src.utils.plan import load_all_plans


def test_build_markdown(tmp_path: Path, plan_factory) -> None:
    plan_factory(
        plan_id="rq1_demo",
        rq="rq1",
        baseline="base1",
        variants=[{"experiment": "var1", "change": "Đổi mô hình.", "fixed": ["split v1"]}],
        path=tmp_path / "plans" / "rq1" / "rq1_demo.yaml",
    )
    plan_factory(
        plan_id="rq2_demo",
        rq="rq2",
        baseline="base2",
        variants=[{"experiment": "var2", "change": "Bỏ augmentation.", "fixed": ["seed"]}],
        path=tmp_path / "plans" / "rq2" / "rq2_demo.yaml",
    )

    plans = load_all_plans(tmp_path / "plans")
    output = tmp_path / "experiment_matrix.md"
    write_matrix(tmp_path / "plans", output)

    text = output.read_text(encoding="utf-8")
    assert "Số plan: 2" in build_markdown(plans)
    assert "# Ma trận thí nghiệm" in text
    assert "Số plan: 2" in text
    assert "Số experiment: 4" in text
    assert "## RQ1 – Model Architecture" in text
    assert "## RQ2 – Data Augmentation & Class Weighting" in text
    assert "| base1 | baseline |" in text
    assert "| var2 | variant |" in text
    assert "Câu hỏi?" in text
