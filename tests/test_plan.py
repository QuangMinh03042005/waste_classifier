"""Test schema và kiểm tra plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.plan import (
    PlanMismatchError,
    PlanValidationError,
    load_all_plans,
    load_plan,
    snapshot_plan,
    validate_for_run,
)


def test_load_valid_plan(plan_factory) -> None:
    path = plan_factory()
    plan = load_plan(path)
    assert plan.plan_id == "rq1_demo"
    assert plan.rq == "rq1"
    assert plan.seeds == [42, 123]
    assert plan.experiments == ["base", "variant"]
    assert len(plan.sha256) == 64


def test_invalid_status(plan_factory) -> None:
    path = plan_factory(status="draft")
    with pytest.raises(PlanValidationError, match="status"):
        load_plan(path)


def test_missing_failure_plan(plan_factory) -> None:
    path = plan_factory(failure_plan="")
    with pytest.raises(PlanValidationError, match="failure_plan"):
        load_plan(path)


def test_duplicate_experiment_inside_plan(plan_factory) -> None:
    path = plan_factory(
        variants=[
            {"experiment": "base", "change": "x", "fixed": ["y"]},
        ]
    )
    with pytest.raises(PlanValidationError, match="trùng"):
        load_plan(path)


def test_duplicate_plan_id(tmp_path: Path, plan_factory) -> None:
    plan_factory(plan_id="same", path=tmp_path / "plans" / "a.yaml")
    plan_factory(plan_id="same", path=tmp_path / "plans" / "b.yaml")
    with pytest.raises(PlanValidationError, match="plan_id trùng"):
        load_all_plans(tmp_path / "plans")


def test_duplicate_experiment_across_plans(tmp_path: Path, plan_factory) -> None:
    plan_factory(
        plan_id="one",
        baseline="shared",
        path=tmp_path / "plans" / "one.yaml",
    )
    plan_factory(
        plan_id="two",
        baseline="shared",
        path=tmp_path / "plans" / "two.yaml",
    )
    with pytest.raises(PlanValidationError, match="experiment trùng"):
        load_all_plans(tmp_path / "plans")


def test_validate_for_run(plan_factory) -> None:
    plan = load_plan(plan_factory())
    validate_for_run(plan, config={"rq": "rq1", "experiment": "variant"}, seed=42)

    with pytest.raises(PlanMismatchError, match="Seed"):
        validate_for_run(plan, config={"rq": "rq1", "experiment": "variant"}, seed=7)
    with pytest.raises(PlanMismatchError, match="không có trong plan"):
        validate_for_run(plan, config={"rq": "rq1", "experiment": "other"}, seed=42)
    with pytest.raises(PlanMismatchError, match="Config thuộc"):
        validate_for_run(plan, config={"rq": "rq2", "experiment": "variant"}, seed=42)


def test_inactive_plan_rejected(plan_factory) -> None:
    plan = load_plan(plan_factory(status="completed"))
    with pytest.raises(PlanMismatchError, match="active"):
        validate_for_run(plan, config={"rq": "rq1", "experiment": "variant"}, seed=42)


def test_snapshot_plan(tmp_path: Path, plan_factory) -> None:
    path = plan_factory()
    plan = load_plan(path)
    run_dir = tmp_path / "run"
    target = snapshot_plan(plan, run_dir)
    assert target.read_bytes() == path.read_bytes()
    assert target.name == "plan.yaml"
