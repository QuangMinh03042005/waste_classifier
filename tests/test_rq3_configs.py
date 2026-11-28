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
CROSS_PLAN = ROOT / "plans" / "rq3" / "cross_dataset_generalization.yaml"
MIX_PLAN = ROOT / "plans" / "rq3" / "source_mix.yaml"
RQ3_SUMMARY = ROOT / "data" / "splits" / "rq3" / "summary.json"
MIX_SUMMARY = ROOT / "data" / "splits" / "rq3" / "mix_summary.json"

CONFIG_META = {
    "rq3_train_realwaste_test_trashnet": {
        "plan": CROSS_PLAN,
        "summary": RQ3_SUMMARY,
        "protocol": "rw_to_tn",
    },
    "rq3_train_trashnet_test_realwaste": {
        "plan": CROSS_PLAN,
        "summary": RQ3_SUMMARY,
        "protocol": "tn_to_rw",
    },
    "rq3_train_pooled_test_each": {
        "plan": CROSS_PLAN,
        "summary": RQ3_SUMMARY,
        "protocol": "pooled",
    },
    "rq3_mix_tn00": {"plan": MIX_PLAN, "summary": MIX_SUMMARY, "protocol": "mix_tn_00"},
    "rq3_mix_tn25": {"plan": MIX_PLAN, "summary": MIX_SUMMARY, "protocol": "mix_tn_25"},
    "rq3_mix_tn50": {"plan": MIX_PLAN, "summary": MIX_SUMMARY, "protocol": "mix_tn_50"},
    "rq3_mix_tn75": {"plan": MIX_PLAN, "summary": MIX_SUMMARY, "protocol": "mix_tn_75"},
}

CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
IDS = [path.stem for path in CONFIGS]


def _summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol_info(path: Path, name: str) -> dict:
    data = _summary(path)
    if "mixes" in data:
        return data["mixes"][name]
    return data["protocols"][name]


def test_moi_config_co_metadata() -> None:
    assert [path.stem for path in CONFIGS] == sorted(CONFIG_META)


@pytest.mark.parametrize("config_path", CONFIGS, ids=IDS)
def test_config_khop_plan_active(config_path: Path) -> None:
    config = load_config(config_path)
    meta = CONFIG_META[config_path.stem]
    plan = load_plan(meta["plan"])

    validate_for_run(plan, config=config, seed=config["seed"])
    assert config["experiment"] in plan.experiments


@pytest.mark.parametrize("config_path", CONFIGS, ids=IDS)
def test_config_giu_bien_co_dinh_cua_plan(config_path: Path) -> None:
    config = load_config(config_path)
    plan = load_plan(CONFIG_META[config_path.stem]["plan"])
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
    meta = CONFIG_META[config_path.stem]
    expected = _protocol_info(meta["summary"], meta["protocol"])

    rows = read_split_csv(ROOT / config["data"]["split_csv"])
    assert dict(Counter(row.split for row in rows)) == expected["counts"]

    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row.split, set()).add(row.dataset)

    if meta["protocol"] == "rw_to_tn":
        assert by_split["train"] == {"realwaste"}
        assert by_split["val"] == {"realwaste"}
        assert by_split["test"] == {"trashnet"}
    elif meta["protocol"] == "tn_to_rw":
        assert by_split["train"] == {"trashnet"}
        assert by_split["val"] == {"trashnet"}
        assert by_split["test"] == {"realwaste"}
    elif meta["protocol"] == "mix_tn_00":
        assert by_split["train"] == {"realwaste"}
        assert by_split["val"] == {"realwaste", "trashnet"}
        assert by_split["test"] == {"realwaste", "trashnet"}
    else:
        assert by_split["train"] == {"realwaste", "trashnet"}
        assert by_split["val"] == {"realwaste", "trashnet"}
        assert by_split["test"] == {"realwaste", "trashnet"}


def test_mix_configs_chi_khac_split_va_ten() -> None:
    mix_paths = sorted(CONFIG_DIR.glob("rq3_mix_tn*.yaml"))
    assert len(mix_paths) == 4

    baseline = load_config(mix_paths[0])
    for path in mix_paths[1:]:
        config = load_config(path)
        left = json.loads(json.dumps(baseline))
        right = json.loads(json.dumps(config))
        for payload in (left, right):
            payload.pop("experiment")
            payload["data"].pop("split_csv")
        assert left == right, f"{path.name} thay đổi quá một biến"


def test_so_luong_train_tang_theo_ti_le() -> None:
    counts = []
    for share in (0, 25, 50, 75):
        rows = read_split_csv(ROOT / f"data/splits/rq3/mix_tn_{share:02d}.csv")
        trashnet_train = sum(
            1 for row in rows if row.dataset == "trashnet" and row.split == "train"
        )
        counts.append(trashnet_train)
    assert counts == sorted(counts)
    assert counts[0] == 0


def test_entrypoint_nhan_config_va_plan() -> None:
    args = build_parser().parse_args(
        ["--config", "configs/rq3/rq3_train_pooled_test_each.yaml", "--plan", str(CROSS_PLAN)]
    )
    assert args.config == Path("configs/rq3/rq3_train_pooled_test_each.yaml")
    assert args.plan == CROSS_PLAN
    assert args.seed is None
