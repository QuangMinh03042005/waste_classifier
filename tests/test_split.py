"""Test tạo split và chống leakage."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from src.data.split import (
    ImageRecord,
    SplitError,
    assign_splits,
    collect_records,
    compute_phash,
    find_groups,
    make_splits,
    validate_ratios,
    validate_split,
)


def make_record(index: int, dataset: str, label: str) -> ImageRecord:
    return ImageRecord(
        path=Path(f"raw/{dataset}/{label}/{index}.jpg"),
        dataset=dataset,
        label=label,
        raw_label=label,
    )


def test_validate_ratios() -> None:
    assert validate_ratios((0.7, 0.15, 0.15)) == {"train": 0.7, "val": 0.15, "test": 0.15}
    with pytest.raises(SplitError, match="Tổng"):
        validate_ratios((0.7, 0.2, 0.2))
    with pytest.raises(SplitError, match="> 0"):
        validate_ratios((1.0, 0.0, 0.0))


def test_find_groups_deduplicates_identical_hashes() -> None:
    hashes = {Path("b.jpg"): 0, Path("a.jpg"): 0, Path("c.jpg"): 0b1111}
    groups = find_groups(hashes, threshold=0)
    assert groups[Path("a.jpg")] == groups[Path("b.jpg")]
    assert groups[Path("c.jpg")] != groups[Path("a.jpg")]
    assert groups[Path("a.jpg")] == "g000001"


def test_assign_splits_group_integrity_and_determinism() -> None:
    records = [
        make_record(index, dataset, label)
        for dataset in ("realwaste", "trashnet")
        for label in ("glass", "paper")
        for index in range(6)
    ]
    group_map = {record.path: f"g{index:03d}" for index, record in enumerate(records)}
    # Hai ảnh liên tiếp là một cặp trùng.
    group_map[records[1].path] = group_map[records[0].path]

    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    first = assign_splits(records, group_map, ratios, seed=42)
    second = assign_splits(records, group_map, ratios, seed=42)
    assert first == second

    assert first[records[0].path] == first[records[1].path]
    assert set(first.values()) == {"train", "val", "test"}
    validate_split(records, first, group_map)

    counts = {split: 0 for split in ("train", "val", "test")}
    for split in first.values():
        counts[split] += 1
    assert counts["train"] > counts["val"]


def test_validate_split_detects_leakage() -> None:
    records = [make_record(0, "realwaste", "glass"), make_record(1, "realwaste", "glass")]
    group_map = {record.path: "g1" for record in records}
    split_map = {records[0].path: "train", records[1].path: "test"}
    with pytest.raises(SplitError, match="group"):
        validate_split(records, split_map, group_map)


def test_compute_phash_is_stable(tmp_path: Path) -> None:
    image = Image.new("L", (32, 32))
    image.putdata([(x * 8 + y) % 256 for x in range(32) for y in range(32)])
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    image.save(first)
    image.save(second)
    assert compute_phash(first) == compute_phash(second)


def test_collect_records_skips_excluded_classes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw = data_root / "raw"
    (raw / "realwaste" / "RealWaste" / "Cardboard").mkdir(parents=True)
    (raw / "realwaste" / "RealWaste" / "Food Organics").mkdir()
    (raw / "trashnet" / "dataset-resized" / "glass").mkdir(parents=True)
    (raw / "trashnet" / "dataset-resized" / "trash").mkdir()

    for path in (
        raw / "realwaste" / "RealWaste" / "Cardboard" / "1.jpg",
        raw / "realwaste" / "RealWaste" / "Food Organics" / "2.jpg",
        raw / "trashnet" / "dataset-resized" / "glass" / "3.jpg",
        raw / "trashnet" / "dataset-resized" / "trash" / "4.jpg",
    ):
        path.write_bytes(b"x")

    records = collect_records(raw)
    assert len(records) == 2
    assert {record.label for record in records} == {"cardboard", "glass"}
    assert all(record.path.as_posix().startswith("raw/") for record in records)


def test_make_splits_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    for dataset, root_name in (("realwaste", "RealWaste"), ("trashnet", "dataset-resized")):
        for label in ("Glass", "Paper") if dataset == "realwaste" else ("glass", "paper"):
            class_dir = raw / dataset / root_name / label
            class_dir.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (8, 8), color=(index * 40, 10, 10)).save(
                    class_dir / f"{index}.jpg"
                )

    output = tmp_path / "splits" / "v1"
    summary = make_splits(raw, output, dedup=False)

    rows = list(csv.DictReader((output / "split.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert summary["counts"]["total"] == 12
    assert (output / "summary.json").is_file()

    with pytest.raises(FileExistsError):
        make_splits(raw, output, dedup=False)
