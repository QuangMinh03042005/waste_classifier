"""Kiểm tra toàn vẹn các split RQ3 đã commit (không đọc ảnh)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits" / "rq3"
FILES = [
    "rw_to_tn.csv",
    "tn_to_rw.csv",
    "pooled.csv",
    "mix_tn_00.csv",
    "mix_tn_25.csv",
    "mix_tn_50.csv",
    "mix_tn_75.csv",
]


def _rows(name: str) -> list[dict[str, str]]:
    with (SPLIT_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("name", FILES)
def test_khong_trung_path_va_khong_leak_group(name: str) -> None:
    rows = _rows(name)

    paths = [row["path"] for row in rows]
    assert len(paths) == len(set(paths)), f"{name}: có đường dẫn trùng"

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    assert not leaked, f"{name}: group nằm nhiều split {next(iter(leaked.items()))}"

    assert {row["split"] for row in rows} <= {"train", "val", "test"}


@pytest.mark.parametrize(
    "name", ["pooled.csv", "mix_tn_00.csv", "mix_tn_25.csv", "mix_tn_50.csv", "mix_tn_75.csv"]
)
def test_test_split_nam_trong_split_v1(name: str) -> None:
    """Các split mix/pooled phải dùng đúng held-out test của split v1."""
    v1_test = {
        row["path"]
        for row in csv.DictReader(
            (ROOT / "data" / "splits" / "v1" / "split.csv").open(encoding="utf-8", newline="")
        )
        if row["split"] == "test"
    }
    test_paths = {row["path"] for row in _rows(name) if row["split"] == "test"}
    assert test_paths <= v1_test, f"{name}: test có ảnh ngoài split v1"


def test_val_test_giong_nhau_giua_cac_mix() -> None:
    baseline = {
        split: {row["path"] for row in _rows("mix_tn_00.csv") if row["split"] == split}
        for split in ("val", "test")
    }
    for name in FILES:
        if name in {"rw_to_tn.csv", "tn_to_rw.csv"}:
            continue
        for split in ("val", "test"):
            paths = {row["path"] for row in _rows(name) if row["split"] == split}
            assert paths == baseline[split], f"{name}: {split} khác mix_tn_00"


def test_mix_tang_dan_va_bat_dau_tu_khong() -> None:
    trashnet_train = []
    for name in ("mix_tn_00.csv", "mix_tn_25.csv", "mix_tn_50.csv", "mix_tn_75.csv", "pooled.csv"):
        trashnet_train.append(
            sum(
                1 for row in _rows(name) if row["dataset"] == "trashnet" and row["split"] == "train"
            )
        )
    assert trashnet_train[0] == 0
    assert trashnet_train == sorted(trashnet_train)
    assert len(set(trashnet_train)) == len(trashnet_train)
