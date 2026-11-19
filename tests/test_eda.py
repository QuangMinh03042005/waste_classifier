"""Test hàm EDA."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.eda import (
    build_summary,
    class_table,
    collect_image_stats,
    duplicate_table,
    quality_issues,
    split_table,
)


def make_stats() -> pd.DataFrame:
    rows = [
        {
            "path": "raw/a.jpg",
            "dataset": "realwaste",
            "label": "glass",
            "split": "train",
            "group_id": "g1",
            "width": 512,
            "height": 512,
            "aspect_ratio": 1.0,
            "min_side": 512,
            "file_size_kb": 100.0,
            "mode": "RGB",
            "format": "JPEG",
            "brightness": 120.0,
            "contrast": 30.0,
            "blur": 100.0,
        },
        {
            "path": "raw/b.jpg",
            "dataset": "trashnet",
            "label": "paper",
            "split": "test",
            "group_id": "g1",
            "width": 512,
            "height": 384,
            "aspect_ratio": 1.33,
            "min_side": 384,
            "file_size_kb": 20.0,
            "mode": "RGB",
            "format": "JPEG",
            "brightness": 225.0,
            "contrast": 10.0,
            "blur": 5.0,
        },
        {
            "path": "raw/c.jpg",
            "dataset": "trashnet",
            "label": "plastic",
            "split": "val",
            "group_id": "g2",
            "width": 512,
            "height": 384,
            "aspect_ratio": 1.33,
            "min_side": 384,
            "file_size_kb": 20.0,
            "mode": "RGB",
            "format": "JPEG",
            "brightness": 150.0,
            "contrast": 20.0,
            "blur": 100.0,
        },
    ]
    return pd.DataFrame(rows)


def test_tables_and_summary() -> None:
    stats = make_stats()
    classes = class_table(stats)
    assert set(classes["label"]) == {"glass", "paper", "plastic"}
    assert classes.loc[classes["label"] == "glass", "realwaste"].item() == 1

    splits = split_table(stats)
    assert set(splits["split"]) == {"train", "test", "val"}

    duplicates = duplicate_table(stats)
    assert len(duplicates) == 1
    assert duplicates.iloc[0]["images"] == 2
    assert "," in duplicates.iloc[0]["labels"]

    issues = quality_issues(stats)
    assert len(issues["label_conflicts"]) == 1
    assert len(issues["too_bright"]) == 1
    assert len(issues["blurry"]) == 1

    summary = build_summary(stats, duplicates)
    assert summary["total"] == 3
    assert summary["by_split"]["train"] == 1
    assert summary["duplicate_groups"] == 1
    assert summary["quality_issues"]["label_conflicts"] == 1
    json.dumps(summary)


def test_collect_image_stats(tmp_path: Path) -> None:
    data = tmp_path / "data"
    records = []
    for index, color in enumerate(((10, 10, 10), (200, 200, 200))):
        path = data / f"raw/{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=color).save(path)
        records.append(
            {
                "path": f"raw/{index}.jpg",
                "dataset": "realwaste",
                "label": "glass",
                "raw_label": "Glass",
                "split": "train",
                "group_id": f"g{index}",
            }
        )
    stats = collect_image_stats(pd.DataFrame(records), data)
    assert len(stats) == 2
    assert stats["width"].tolist() == [64, 64]
    assert stats["height"].tolist() == [48, 48]
    assert stats["brightness"].iloc[0] < stats["brightness"].iloc[1]
