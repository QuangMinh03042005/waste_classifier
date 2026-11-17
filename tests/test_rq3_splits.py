"""Test sinh split RQ3."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.rq3_splits import make_cross_dataset_split, make_rq3_splits
from src.data.split import SplitRow, read_split_csv, write_split_csv


def row(path: str, dataset: str, split: str, group: str, label: str = "glass") -> SplitRow:
    return SplitRow(
        path=path,
        dataset=dataset,
        label=label,
        raw_label=label.capitalize(),
        split=split,
        group_id=group,
    )


def test_cross_dataset_split_excludes_mixed_groups(tmp_path: Path) -> None:
    rows = [
        row("raw/rw/train.jpg", "realwaste", "train", "g1"),
        row("raw/rw/val.jpg", "realwaste", "val", "g2"),
        row("raw/rw/test.jpg", "realwaste", "test", "g3"),
        row("raw/tn/a.jpg", "trashnet", "train", "g4"),
        row("raw/tn/b.jpg", "trashnet", "test", "g5"),
        row("raw/tn/mixed.jpg", "trashnet", "train", "g_mixed"),
        row("raw/rw/mixed.jpg", "realwaste", "train", "g_mixed"),
    ]
    split_rows, excluded = make_cross_dataset_split(
        rows, train_dataset="realwaste", test_dataset="trashnet"
    )

    assert "g_mixed" in excluded
    assert {item.path for item in split_rows} == {
        "raw/rw/train.jpg",
        "raw/rw/val.jpg",
        "raw/tn/a.jpg",
        "raw/tn/b.jpg",
    }
    splits = {item.path: item.split for item in split_rows}
    assert splits["raw/rw/train.jpg"] == "train"
    assert splits["raw/rw/val.jpg"] == "val"
    assert splits["raw/tn/a.jpg"] == "test"
    assert splits["raw/tn/b.jpg"] == "test"


def test_make_rq3_splits(tmp_path: Path) -> None:
    rows = [
        row("raw/rw/train.jpg", "realwaste", "train", "g1"),
        row("raw/rw/val.jpg", "realwaste", "val", "g2"),
        row("raw/tn/a.jpg", "trashnet", "train", "g3"),
        row("raw/tn/b.jpg", "trashnet", "test", "g4"),
    ]
    source = tmp_path / "split.csv"
    write_split_csv(rows, source)

    output = tmp_path / "rq3"
    summary = make_rq3_splits(source, output)

    assert summary["protocols"]["rw_to_tn"]["counts"] == {"test": 2, "train": 1, "val": 1}
    assert (output / "rw_to_tn.csv").is_file()
    assert (output / "tn_to_rw.csv").is_file()
    assert (output / "pooled.csv").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))

    rw_to_tn = read_split_csv(output / "rw_to_tn.csv")
    assert {item.dataset for item in rw_to_tn if item.split == "test"} == {"trashnet"}
