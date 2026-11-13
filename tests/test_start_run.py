"""Test tích hợp plan vào lúc bắt đầu run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.run import start_run
from src.utils.plan import PlanMismatchError, PlanRequiredError


def make_config(rq: str = "rq1", experiment: str = "variant") -> dict:
    return {"rq": rq, "experiment": experiment, "seed": 42}


def test_requires_plan(tmp_path: Path) -> None:
    with pytest.raises(PlanRequiredError, match="--plan"):
        start_run(
            config=make_config(),
            run_id="run_s42",
            seed=42,
            experiments_root=tmp_path / "experiments",
        )


def test_unplanned_run_is_marked(tmp_path: Path, plan_factory) -> None:
    logger, plan = start_run(
        config=make_config(),
        run_id="debug_s42",
        seed=42,
        rq="rq1",
        experiments_root=tmp_path / "experiments",
        allow_unplanned=True,
    )
    assert plan is None
    run = json.loads(logger.run_json.read_text(encoding="utf-8"))
    assert run["plan"] == {"status": "unplanned"}


def test_start_run_snapshots_plan(tmp_path: Path, plan_factory) -> None:
    path = plan_factory()
    logger, plan = start_run(
        config=make_config(),
        run_id="run_s42",
        seed=42,
        plan_path=path,
        experiments_root=tmp_path / "experiments",
    )

    assert plan is not None
    snapshot = logger.root / "plan.yaml"
    assert snapshot.read_bytes() == path.read_bytes()

    run = json.loads(logger.run_json.read_text(encoding="utf-8"))
    assert run["plan"]["plan_id"] == "rq1_demo"
    assert run["plan"]["sha256"] == plan.sha256
    assert run["plan"]["snapshot"] == "plan.yaml"


def test_resume_rejects_changed_plan(tmp_path: Path, plan_factory) -> None:
    path = plan_factory()
    start_run(
        config=make_config(),
        run_id="run_s42",
        seed=42,
        plan_path=path,
        experiments_root=tmp_path / "experiments",
    )

    path.write_text(path.read_text(encoding="utf-8") + "\nnotes: changed\n", encoding="utf-8")

    with pytest.raises(PlanMismatchError, match="Plan đã thay đổi"):
        start_run(
            config=make_config(),
            run_id="run_s42",
            seed=42,
            plan_path=path,
            experiments_root=tmp_path / "experiments",
            resume=True,
        )


def test_wrong_seed_rejected(tmp_path: Path, plan_factory) -> None:
    path = plan_factory()
    with pytest.raises(PlanMismatchError, match="Seed"):
        start_run(
            config=make_config(),
            run_id="run_s99",
            seed=99,
            plan_path=path,
            experiments_root=tmp_path / "experiments",
        )
