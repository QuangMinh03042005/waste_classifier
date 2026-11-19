"""Chuẩn bị external test GD (Garbage Dataset) cho RQ1.

Chỉ dùng 5 lớp: cardboard, glass, metal, paper, plastic.
Không sửa split v1; external test được lưu riêng.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.download import IMAGE_SUFFIXES, download_file, safe_extract
from src.data.split import ImageRecord, SplitRow, compute_phash, find_groups, write_split_csv

GD_URL = "https://www.kaggle.com/api/v1/datasets/download/sumn2u/garbage-classification-v2"
GD_LICENSE = "MIT"
GD_CITATION = (
    "Kunwar, S. The Garbage Dataset (GD): A Multi-Class Image Benchmark "
    "for Automated Waste Segregation."
)

GD_MAPPING: dict[str, str | None] = {
    "cardboard": "cardboard",
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "biological": None,
    "battery": None,
    "trash": None,
    "shoes": None,
    "clothes": None,
}

GD_EXPECTED_DIRS = {"cardboard", "glass", "metal", "paper", "plastic"}


@dataclass(frozen=True, slots=True)
class ExternalRecord:
    path: Path
    dataset: str
    label: str
    raw_label: str
    group_id: str = ""


def canonical_external_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if key not in GD_MAPPING:
        raise ValueError(f"Nhãn GD lạ: {raw_label!r}")
    return GD_MAPPING[key]


def _find_root(extract_dir: Path) -> Path:
    for path in sorted(extract_dir.rglob("*")):
        if not path.is_dir():
            continue
        children = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        if GD_EXPECTED_DIRS <= children:
            return path
    raise RuntimeError(f"Không tìm thấy thư mục lớp GD trong {extract_dir}")


def download_gd(data_root: Path, *, force: bool = False) -> Path:
    """Tải và giải nén GD vào data/raw/external/gd/<root>."""
    external_root = data_root / "external" / "gd"
    if external_root.exists() and any(external_root.iterdir()) and not force:
        roots = [path for path in external_root.iterdir() if path.is_dir()]
        if roots:
            return roots[0]

    archive = data_root / "external" / "_downloads" / "gd.zip"
    if force:
        archive.unlink(missing_ok=True)
    if not archive.is_file() or archive.stat().st_size == 0:
        download_file(GD_URL, archive)

    temp = data_root / "external" / "_tmp"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True)
    try:
        safe_extract(archive, temp)
        root = _find_root(temp)
        if external_root.exists():
            shutil.rmtree(external_root)
        external_root.mkdir(parents=True)
        shutil.move(str(root), str(external_root / root.name))
        return external_root / root.name
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def collect_records(gd_root: Path, data_root: Path) -> list[ExternalRecord]:
    records: list[ExternalRecord] = []
    for class_dir in sorted(path for path in gd_root.iterdir() if path.is_dir()):
        label = canonical_external_label(class_dir.name)
        if label is None:
            continue
        for image in sorted(class_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    ExternalRecord(
                        path=image.relative_to(data_root),
                        dataset="gd",
                        label=label,
                        raw_label=class_dir.name,
                    )
                )
    if not records:
        raise RuntimeError(f"Không quét được ảnh GD trong {gd_root}")
    return records


def _train_hashes(split_csv: Path, data_root: Path) -> np.ndarray:
    rows = pd.read_csv(split_csv)
    values = [compute_phash(data_root / row["path"]) for _, row in tqdm(rows.iterrows(), desc="Train pHash")]
    return np.array(values, dtype=np.uint64)


def dedup_external(
    records: list[ExternalRecord],
    *,
    data_root: Path,
    train_split: Path,
    threshold: int = 5,
) -> tuple[list[ExternalRecord], dict[str, Any]]:
    """Loại ảnh trùng với train và trùng trong nội bộ external."""
    hashes = {
        record.path: compute_phash(data_root / record.path)
        for record in tqdm(records, desc="GD pHash")
    }
    train_values = _train_hashes(train_split, data_root)

    kept: list[ExternalRecord] = []
    overlap = 0
    for record in records:
        value = hashes[record.path]
        distances = np.bitwise_count(train_values ^ np.uint64(value))
        if int(distances.min()) <= threshold:
            overlap += 1
            continue
        kept.append(record)

    # Gom ảnh trùng trong external, giữ một đại diện mỗi group.
    groups = find_groups({record.path: hashes[record.path] for record in kept}, threshold)
    representatives: dict[str, ExternalRecord] = {}
    internal_duplicates = 0
    for record in kept:
        group = groups[record.path]
        if group in representatives:
            internal_duplicates += 1
            continue
        representatives[group] = ExternalRecord(
            path=record.path,
            dataset=record.dataset,
            label=record.label,
            raw_label=record.raw_label,
            group_id=group,
        )

    final = sorted(representatives.values(), key=lambda item: item.path.as_posix())
    summary = {
        "source": GD_URL,
        "license": GD_LICENSE,
        "citation": GD_CITATION,
        "original_records": len(records),
        "excluded_train_overlap": overlap,
        "excluded_internal_duplicates": internal_duplicates,
        "kept": len(final),
        "threshold": threshold,
        "counts": dict(sorted(Counter(record.label for record in final).items())),
    }
    return final, summary


def write_external_csv(records: list[ExternalRecord], output: Path) -> Path:
    rows = [
        SplitRow(
            path=record.path.as_posix(),
            dataset=record.dataset,
            label=record.label,
            raw_label=record.raw_label,
            split="test",
            group_id=record.group_id,
        )
        for record in records
    ]
    write_split_csv(rows, output)
    return output


def prepare_external_gd(
    *,
    data_root: Path,
    train_split: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    gd_root = download_gd(data_root, force=force)
    records = collect_records(gd_root, data_root)
    kept, summary = dedup_external(records, data_root=data_root, train_split=train_split)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = write_external_csv(kept, output_dir / "gd_test.csv")
    summary.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "gd_root": str(gd_root),
            "csv": str(output_csv),
        }
    )
    (output_dir / "gd_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
