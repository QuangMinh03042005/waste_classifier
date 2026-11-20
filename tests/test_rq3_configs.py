"""Test config RQ3 khớp plan đã đăng ký và đúng protocol split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from research.rq3.train import build_parser
from src.data.split import read_split_csv
from src.utils.config import load_config
from src.utils.plan import load_plan, validate_for_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "rq3"
PLAN_PATH = ROOT / "plans" / "rq3" / "cross_dataset_generalization.yaml"
SUMMARY_PATH = ROOT / "data" / "splits" / "rq3" / "summary.json"

PROTOCOLS = {
    "rq3_train_realwaste_test_trashnet": "rw_to_tn",
    "rq3_train_trashnet_test_realwaste": "tn_to_rw",
    "rq3_train_pooled_test_each": "pooled",
}

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
IDS = [path.stem for path in CONFIGS]


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_du_ba_config_cho_ba_protocol() -> None:
    assert [path.stem for path in CONFIGS] == sorted(PROTOCOLS)


@pytest.mark.parametrize("config_path", CONFIGS, ids=IDS)
def test_config_khop_plan_active(config_path: Path) -> None:
    config = load_config(config_path)
    plan = load_plan(PLAN_PATH)

    validate_for_run(plan, config=config, seed=config["seed"])
    assert config["experiment"] in PROTOCOLS


@pytest.mark.parametrize("config_path", CONFIGS, ids=IDS)
def test_config_giu_bien_co_dinh_cua_plan(config_path: Path) -> None:
    config = load_config(config_path)
    plan = load_plan(PLAN_PATH)
    budget = plan.raw["budget"]

    assert config["model"]["name"] == "mobilenet_v3_small"
    assert config["model"]["pretrained"] is True
    assert config["data"]["augmentation"] is False
    assert config["data"]["class_weighting"] is False
    assert config["data"]["batch_size"] == budget["batch_size"]
    assert config["training"]["epochs"] <= budget["max_epochs"]
    assert config["training"]["early_stopping_patience"] == 7
    assert config["training"]["optimizer"] == "adamw"
    assert config["seed"] in plan.seeds


@pytest.mark.parametrize("config_path", CONFIGS, ids=IDS)
def test_split_khop_protocol(config_path: Path) -> None:
    config = load_config(config_path)
    protocol = PROTOCOLS[config["experiment"]]
    expected = _summary()["protocols"][protocol]

    rows = read_split_csv(ROOT / config["data"]["split_csv"])
    assert dict(Counter(row.split for row in rows)) == expected["counts"]

    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row.split, set()).add(row.dataset)

    if protocol == "rw_to_tn":
        assert by_split["train"] == {"realwaste"}
        assert by_split["val"] == {"realwaste"}
        assert by_split["test"] == {"trashnet"}
    elif protocol == "tn_to_rw":
        assert by_split["train"] == {"trashnet"}
        assert by_split["val"] == {"trashnet"}
        assert by_split["test"] == {"realwaste"}
    else:
        assert all(datasets == {"realwaste", "trashnet"} for datasets in by_split.values())


def test_entrypoint_nhan_config_va_plan() -> None:
    args = build_parser().parse_args(
        ["--config", "configs/rq3/rq3_train_pooled_test_each.yaml", "--plan", str(PLAN_PATH)]
    )
    assert args.config == Path("configs/rq3/rq3_train_pooled_test_each.yaml")
    assert args.plan == PLAN_PATH
    assert args.seed is None
