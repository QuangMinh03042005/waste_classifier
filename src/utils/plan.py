"""Đọc, kiểm tra và chụp lại kế hoạch thí nghiệm (pre-registration).

Mỗi plan mô tả một RQ với baseline + các biến thể. Plan phải được commit và
được kiểm tra trước khi training; bản copy được lưu vào thư mục run.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0"
VALID_RQ = {"rq1", "rq2", "rq3"}
VALID_STATUS = {"active", "completed", "aborted"}
_PLAN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PlanError(ValueError):
    """Lỗi chung của plan."""


class PlanValidationError(PlanError):
    """Plan sai schema."""


class PlanRequiredError(PlanError):
    """Training thiếu plan."""


class PlanMismatchError(PlanError):
    """Plan không khớp config/seed/rq."""


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def plan_id(self) -> str:
        return str(self.raw["plan_id"])

    @property
    def rq(self) -> str:
        return str(self.raw["rq"])

    @property
    def status(self) -> str:
        return str(self.raw["status"])

    @property
    def seeds(self) -> list[int]:
        return [int(seed) for seed in self.raw["seeds"]]

    @property
    def experiments(self) -> list[str]:
        baseline = str(self.raw["baseline"]["experiment"])
        variants = [str(item["experiment"]) for item in self.raw["variants"]]
        return [baseline, *variants]

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw["data"])


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require(data: dict[str, Any], key: str, expected: type, context: str) -> Any:
    if key not in data:
        raise PlanValidationError(f"{context}: thiếu trường '{key}'")
    value = data[key]
    if not isinstance(value, expected):
        raise PlanValidationError(
            f"{context}: '{key}' phải là {expected.__name__}, nhận {type(value).__name__}"
        )
    return value


def _require_str(data: dict[str, Any], key: str, context: str, *, allow_empty: bool = False) -> str:
    value = _require(data, key, str, context)
    if not allow_empty and not value.strip():
        raise PlanValidationError(f"{context}: '{key}' không được rỗng")
    return value


def validate_plan_data(data: Any, *, context: str = "plan") -> dict[str, Any]:
    """Kiểm tra schema plan, trả lại chính dict đó."""
    if not isinstance(data, dict):
        raise PlanValidationError(f"{context}: nội dung phải là mapping")

    schema_version = _require_str(data, "schema_version", context)
    if schema_version != SCHEMA_VERSION:
        raise PlanValidationError(
            f"{context}: schema_version phải là {SCHEMA_VERSION!r}, nhận {schema_version!r}"
        )

    plan_id = _require_str(data, "plan_id", context)
    if not _PLAN_ID_RE.match(plan_id):
        raise PlanValidationError(f"{context}: plan_id không hợp lệ: {plan_id!r}")

    rq = _require_str(data, "rq", context)
    if rq not in VALID_RQ:
        raise PlanValidationError(f"{context}: rq phải thuộc {sorted(VALID_RQ)}, nhận {rq!r}")

    status = _require_str(data, "status", context)
    if status not in VALID_STATUS:
        raise PlanValidationError(
            f"{context}: status phải thuộc {sorted(VALID_STATUS)}, nhận {status!r}"
        )

    for key in ("title", "author", "created_at", "question", "hypothesis"):
        _require_str(data, key, context)
    _require_str(data, "expected_result", context)
    _require_str(data, "failure_plan", context)

    seeds = _require(data, "seeds", list, context)
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise PlanValidationError(f"{context}: 'seeds' phải là danh sách số nguyên khác rỗng")
    if len(set(seeds)) != len(seeds):
        raise PlanValidationError(f"{context}: 'seeds' bị trùng")

    baseline = _require(data, "baseline", dict, context)
    _require_str(baseline, "experiment", f"{context}.baseline")
    _require_str(baseline, "description", f"{context}.baseline")

    variants = _require(data, "variants", list, context)
    if not variants:
        raise PlanValidationError(f"{context}: cần ít nhất một biến thể")
    for index, variant in enumerate(variants):
        item_context = f"{context}.variants[{index}]"
        if not isinstance(variant, dict):
            raise PlanValidationError(f"{item_context}: phải là mapping")
        _require_str(variant, "experiment", item_context)
        _require_str(variant, "change", item_context)
        fixed = _require(variant, "fixed", list, item_context)
        if not fixed or any(not isinstance(item, str) or not item.strip() for item in fixed):
            raise PlanValidationError(f"{item_context}: 'fixed' phải là danh sách chuỗi khác rỗng")

    experiments = [baseline["experiment"], *[item["experiment"] for item in variants]]
    if len(set(experiments)) != len(experiments):
        raise PlanValidationError(f"{context}: tên experiment bị trùng trong plan")

    comparison = _require(data, "comparison", dict, context)
    for key in ("metric", "test_policy", "success_criterion"):
        _require_str(comparison, key, f"{context}.comparison")

    dataset = _require(data, "data", dict, context)
    _require_str(dataset, "splits_version", f"{context}.data")
    class_names = _require(dataset, "class_names", list, f"{context}.data")
    invalid_names = any(
        not isinstance(name, str) or not name.strip() for name in class_names
    )
    if not class_names or invalid_names:
        raise PlanValidationError(
            f"{context}.data: 'class_names' phải là danh sách chuỗi khác rỗng"
        )
    if len(set(class_names)) != len(class_names):
        raise PlanValidationError(f"{context}.data: 'class_names' bị trùng")

    if "budget" in data and not isinstance(data["budget"], dict):
        raise PlanValidationError(f"{context}: 'budget' phải là mapping")

    return data


def load_plan(path: Path) -> ExperimentPlan:
    """Đọc và kiểm tra một file plan YAML."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy plan: {path}")
    content = path.read_bytes()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise PlanValidationError(f"{path}: YAML lỗi: {exc}") from exc

    validate_plan_data(data, context=str(path))
    return ExperimentPlan(path=path, raw=data, sha256=sha256_bytes(content))


def load_all_plans(plans_root: Path) -> list[ExperimentPlan]:
    """Đọc tất cả plan, kiểm tra trùng plan_id và trùng tên experiment."""
    plans_root = Path(plans_root)
    plans: list[ExperimentPlan] = []
    seen_plan_ids: dict[str, Path] = {}
    seen_experiments: dict[str, Path] = {}

    for path in sorted(plans_root.rglob("*.yaml")):
        plan = load_plan(path)
        if plan.plan_id in seen_plan_ids:
            raise PlanValidationError(
                f"plan_id trùng '{plan.plan_id}': {seen_plan_ids[plan.plan_id]} và {path}"
            )
        seen_plan_ids[plan.plan_id] = path
        for experiment in plan.experiments:
            if experiment in seen_experiments:
                raise PlanValidationError(
                    f"experiment trùng '{experiment}': {seen_experiments[experiment]} và {path}"
                )
            seen_experiments[experiment] = path
        plans.append(plan)

    return plans


def validate_for_run(
    plan: ExperimentPlan,
    *,
    config: dict[str, Any],
    seed: int,
    rq: str | None = None,
) -> None:
    """Đảm bảo plan khớp rq, experiment và seed sắp chạy."""
    if plan.status != "active":
        raise PlanMismatchError(
            f"Plan {plan.plan_id} có status {plan.status!r}; chỉ chạy khi 'active'."
        )
    if rq is not None and rq != plan.rq:
        raise PlanMismatchError(f"Plan thuộc {plan.rq}, không phải {rq}.")
    config_rq = config.get("rq")
    if config_rq is not None and config_rq != plan.rq:
        raise PlanMismatchError(f"Config thuộc {config_rq}, plan thuộc {plan.rq}.")

    experiment = config.get("experiment")
    if not experiment:
        raise PlanMismatchError("Config thiếu 'experiment'; không đối chiếu được với plan.")
    if experiment not in plan.experiments:
        raise PlanMismatchError(
            f"Experiment {experiment!r} không có trong plan {plan.plan_id}. "
            f"Hợp lệ: {', '.join(plan.experiments)}"
        )
    if seed not in plan.seeds:
        raise PlanMismatchError(
            f"Seed {seed} không có trong plan {plan.plan_id} (hợp lệ: {plan.seeds})."
        )


def plan_info(plan: ExperimentPlan) -> dict[str, Any]:
    """Thông tin ghi vào run.json."""
    return {
        "plan_id": plan.plan_id,
        "rq": plan.rq,
        "status": plan.status,
        "sha256": plan.sha256,
        "path": str(plan.path),
        "snapshot": "plan.yaml",
    }


def snapshot_plan(plan: ExperimentPlan, run_dir: Path) -> Path:
    """Chép nguyên bản plan vào thư mục run và kiểm tra hash."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "plan.yaml"
    temp = target.with_name(target.name + ".tmp")
    content = plan.path.read_bytes()
    temp.write_bytes(content)
    os.replace(temp, target)

    if sha256_bytes(target.read_bytes()) != plan.sha256:
        raise PlanMismatchError(f"Plan snapshot không khớp hash gốc: {plan.path}")
    return target
