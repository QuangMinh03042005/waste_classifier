"""Fixture dùng chung cho test plan."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def plan_dict(
    *,
    plan_id: str = "rq1_demo",
    rq: str = "rq1",
    status: str = "active",
    baseline: str = "base",
    variants: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "plan_id": plan_id,
        "rq": rq,
        "title": "Demo plan",
        "author": "Tester",
        "created_at": "2026-11-13",
        "status": status,
        "question": "Câu hỏi?",
        "hypothesis": "Giả thuyết.",
        "expected_result": "Kỳ vọng.",
        "failure_plan": "Nếu sai thì phân tích.",
        "baseline": {"experiment": baseline, "description": "Baseline."},
        "variants": variants
        or [{"experiment": "variant", "change": "Thay đổi.", "fixed": ["split v1"]}],
        "comparison": {
            "metric": "macro_f1",
            "test_policy": "Chỉ test một lần.",
            "success_criterion": "Cao hơn baseline.",
        },
        "data": {"splits_version": "v1", "class_names": ["a", "b"]},
        "seeds": [42, 123],
    }


@pytest.fixture
def plan_factory(tmp_path: Path):
    def make(
        *,
        plan_id: str = "rq1_demo",
        rq: str = "rq1",
        status: str = "active",
        baseline: str = "base",
        variants: list[dict] | None = None,
        path: Path | None = None,
        **overrides,
    ) -> Path:
        data = plan_dict(
            plan_id=plan_id,
            rq=rq,
            status=status,
            baseline=baseline,
            variants=variants,
        )
        data.update(overrides)
        target = path or (tmp_path / "plans" / rq / f"{plan_id}.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return target

    return make
