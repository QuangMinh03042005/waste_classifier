"""Test cấu hình RQ2: chỉ khác biến ablation và khớp plan.

Mục đích: bảo vệ quy tắc "mỗi biến thể chỉ thay một biến" của ablation study.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.plan import load_plan, validate_for_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "rq2"
PLAN_PATH = ROOT / "plans" / "rq2" / "augmentation_ablation.yaml"

# experiment -> (augmentation, class_weighting)
ARMS = {
    "rq2_mobilenetv3s_full": (True, True),
    "rq2_mobilenetv3s_noaug": (False, True),
    "rq2_mobilenetv3s_nocw": (True, False),
}


def _diff_keys(left: dict, right: dict, prefix: str = "") -> set[str]:
    """Liệt kê đường dẫn khoá khác nhau giữa hai mapping lồng nhau."""
    diffs: set[str] = set()
    for key in set(left) | set(right):
        path = f"{prefix}.{key}" if prefix else key
        if key not in left or key not in right:
            diffs.add(path)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            diffs |= _diff_keys(left[key], right[key], path)
        elif left[key] != right[key]:
            diffs.add(path)
    return diffs


@pytest.mark.parametrize("experiment", sorted(ARMS))
def test_arm_config_matches_design(experiment: str) -> None:
    config = load_config(CONFIG_DIR / f"{experiment}.yaml")

    augmentation, class_weighting = ARMS[experiment]
    assert config["experiment"] == experiment
    assert config["rq"] == "rq2"
    assert config["data"]["augmentation"] is augmentation
    assert config["data"]["class_weighting"] is class_weighting
    assert config["training"]["epochs"] == 30
    assert config["training"]["early_stopping_patience"] == 7


def test_arms_differ_only_in_ablation_flags() -> None:
    base = load_config(CONFIG_DIR / "rq2_mobilenetv3s_full.yaml")
    expected = {
        "rq2_mobilenetv3s_noaug": {"experiment", "data.augmentation"},
        "rq2_mobilenetv3s_nocw": {"experiment", "data.class_weighting"},
    }

    for experiment, keys in expected.items():
        other = load_config(CONFIG_DIR / f"{experiment}.yaml")
        assert _diff_keys(base, other) == keys


def test_configs_match_active_plan() -> None:
    plan = load_plan(PLAN_PATH)

    assert plan.status == "active"
    assert plan.rq == "rq2"
    assert plan.seeds == [42, 123, 2024]
    assert set(plan.experiments) == set(ARMS)

    for experiment in ARMS:
        config = load_config(CONFIG_DIR / f"{experiment}.yaml")
        for seed in plan.seeds:
            validate_for_run(plan, config=config, seed=seed, rq="rq2")


def test_smoke_config_does_not_evaluate_test() -> None:
    plan = load_plan(PLAN_PATH)
    config = load_config(CONFIG_DIR / "debug" / "smoke_full.yaml")

    assert config["experiment"] in plan.experiments
    assert config["data"]["augmentation"] is True
    assert config["data"]["class_weighting"] is True
    assert config["training"]["epochs"] == 1
    assert config["training"]["evaluate_test"] is False
    validate_for_run(plan, config=config, seed=42, rq="rq2")
