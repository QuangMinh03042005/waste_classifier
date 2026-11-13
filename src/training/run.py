"""Khởi tạo một run training có kiểm tra kế hoạch thí nghiệm.

Mọi entrypoint training phải gọi `start_run` thay vì tạo `RunLogger` trực tiếp,
để đảm bảo plan đã được commit và khớp config/seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.plan import (
    ExperimentPlan,
    PlanMismatchError,
    PlanRequiredError,
    load_plan,
    plan_info,
    snapshot_plan,
    validate_for_run,
)
from src.utils.run_logger import RunLogger


def start_run(
    *,
    config: dict[str, Any],
    run_id: str,
    seed: int,
    plan_path: Path | None = None,
    rq: str | None = None,
    experiments_root: Path | None = None,
    resume: bool = False,
    from_epoch: int | None = None,
    allow_unplanned: bool = False,
) -> tuple[RunLogger, ExperimentPlan | None]:
    """Tạo run sau khi kiểm tra plan; trả về (logger, plan)."""
    plan: ExperimentPlan | None = None

    if plan_path is None:
        if not allow_unplanned:
            raise PlanRequiredError(
                "Thiếu --plan. Mọi run phải có kế hoạch thí nghiệm đã commit "
                "(hoặc dùng allow_unplanned=True cho debug)."
            )
        if rq is None:
            raise ValueError("Chạy unplanned vẫn phải truyền rq.")
    else:
        plan = load_plan(plan_path)
        validate_for_run(plan, config=config, seed=seed, rq=rq)
        rq = plan.rq

    if rq is None:
        raise ValueError("Không xác định được rq.")

    logger = RunLogger(
        rq,
        run_id,
        config,
        seed=seed,
        experiments_root=experiments_root,
        resume=resume,
        from_epoch=from_epoch,
    )

    if plan is None:
        logger.update(plan={"status": "unplanned"})
        return logger, plan

    if resume:
        previous = json.loads(logger.run_json.read_text(encoding="utf-8"))
        old_hash = (previous.get("plan") or {}).get("sha256")
        if old_hash and old_hash != plan.sha256:
            raise PlanMismatchError(
                "Plan đã thay đổi so với lần chạy trước; không resume với plan khác. "
                "Tạo run mới hoặc khôi phục đúng plan."
            )

    snapshot_plan(plan, logger.root)
    logger.update(plan=plan_info(plan))
    return logger, plan
