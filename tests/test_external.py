"""Test chuẩn bị external test GD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import external
from src.data.external import (
    ExternalRecord,
    canonical_external_label,
    collect_records,
    dedup_external,
    write_external_csv,
)


def test_mapping() -> None:
    assert canonical_external_label("Cardboard") == "cardboard"
    assert canonical_external_label("plastic") == "plastic"
    assert canonical_external_label("biological") is None
    with pytest.raises(ValueError, match="lạ"):
        canonical_external_label("unknown")


def test_collect_records(tmp_path) -> None:
    data_root = tmp_path / "data"
    gd_root = data_root / "external" / "gd" / "garbage-dataset"
    for label in ("cardboard", "glass", "metal", "paper", "plastic", "biological"):
        folder = gd_root / label
        folder.mkdir(parents=True)
        (folder / "x.jpg").write_bytes(b"x")
        (folder / "ignore.txt").write_text("x")

    records = collect_records(gd_root, data_root)
    assert len(records) == 5
    assert {record.label for record in records} == {
        "cardboard",
        "glass",
        "metal",
        "paper",
        "plastic",
    }
    assert all(record.path.as_posix().startswith("external/") for record in records)


def test_dedup_and_csv(tmp_path, monkeypatch) -> None:
    records = [
        ExternalRecord(path=Path("external/a.jpg"), dataset="gd", label="glass", raw_label="glass"),
        ExternalRecord(path=Path("external/b.jpg"), dataset="gd", label="metal", raw_label="metal"),
        ExternalRecord(path=Path("external/c.jpg"), dataset="gd", label="paper", raw_label="paper"),
        ExternalRecord(
            path=Path("external/d.jpg"),
            dataset="gd",
            label="plastic",
            raw_label="plastic",
        ),
    ]
    hashes = {
        "a.jpg": 1,
        "b.jpg": 1,
        "c.jpg": 65535,
        "d.jpg": 65535,
    }

    monkeypatch.setattr(external, "compute_phash", lambda path: hashes[path.name])
    monkeypatch.setattr(
        external, "_train_hashes", lambda split, root: np.array([1], dtype=np.uint64)
    )

    kept, summary = dedup_external(
        records,
        data_root=tmp_path,
        train_split=tmp_path / "split.csv",
        threshold=5,
    )
    # a và b trùng train (hash 1) -> loại; c và d trùng nhau -> giữ 1.
    assert len(kept) == 1
    assert kept[0].path in {Path("external/c.jpg"), Path("external/d.jpg")}
    assert summary["excluded_train_overlap"] == 2
    assert summary["excluded_internal_duplicates"] == 1

    output = write_external_csv(kept, tmp_path / "gd_test.csv")
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.iloc[0]["split"] == "test"
    assert frame.iloc[0]["dataset"] == "gd"
