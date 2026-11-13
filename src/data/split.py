"""Tạo split train/val/test có kiểm soát trùng lặp.

Pipeline:
1. Quét ảnh thuộc 5 lớp chung.
2. Tính perceptual hash (pHash) để gom ảnh trùng/near-duplicate thành group.
3. Chia theo group, stratified theo (dataset, class), tỷ lệ 70/15/15.
4. Ghi `split.csv` + `summary.json` để tái lập.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.fft import dct

from src.data.download import IMAGE_SUFFIXES
from src.data.labels import CANONICAL_CLASSES, canonical_label
from src.data.manifest import DATASETS

SCHEMA_VERSION = "1.0"
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_PHASH_THRESHOLD = 5


class SplitError(RuntimeError):
    """Lỗi khi tạo hoặc kiểm tra split."""


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path  # tương đối so với data root, ví dụ raw/realwaste/...
    dataset: str
    label: str
    raw_label: str


@dataclass(frozen=True, slots=True)
class SplitRow:
    path: str
    dataset: str
    label: str
    raw_label: str
    split: str
    group_id: str


def dataset_root(raw_root: Path, dataset: str) -> Path:
    """Tìm thư mục gốc dữ liệu của dataset (ưu tiên MANIFEST.json)."""
    dataset_dir = raw_root / dataset
    manifest_path = dataset_dir / "MANIFEST.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        root_name = manifest.get("root_name")
        if root_name and (dataset_dir / root_name).is_dir():
            return dataset_dir / root_name

    fallback = {"realwaste": "RealWaste", "trashnet": "dataset-resized"}.get(dataset)
    if fallback and (dataset_dir / fallback).is_dir():
        return dataset_dir / fallback

    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu cho {dataset} trong {raw_root}. "
        "Chạy `python -m src.data.download --dataset all` trước."
    )


def collect_records(raw_root: Path) -> list[ImageRecord]:
    """Quét toàn bộ ảnh thuộc 5 lớp chung, bỏ qua lớp bị loại."""
    raw_root = Path(raw_root).resolve()
    data_root = raw_root.parent
    records: list[ImageRecord] = []

    for dataset in DATASETS:
        root = dataset_root(raw_root, dataset)
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if class_dir.name.startswith("."):
                continue
            label = canonical_label(dataset, class_dir.name)
            if label is None:
                continue
            for image in sorted(class_dir.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                    records.append(
                        ImageRecord(
                            path=image.relative_to(data_root),
                            dataset=dataset,
                            label=label,
                            raw_label=class_dir.name,
                        )
                    )

    if not records:
        raise SplitError(f"Không quét được ảnh nào trong {raw_root}")
    return records


def compute_phash(path: Path, *, hash_size: int = 8, factor: int = 4) -> int:
    """pHash 64-bit: grayscale 32x32, DCT 2D, so với median."""
    image_size = hash_size * factor
    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)

    coefficients = dct(dct(pixels, axis=1, norm="ortho"), axis=0, norm="ortho")
    low = coefficients[:hash_size, :hash_size].flatten()
    median = float(np.median(low[1:]))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def find_groups(hashes: dict[Path, int], threshold: int = DEFAULT_PHASH_THRESHOLD) -> dict[Path, str]:
    """Gom các ảnh có Hamming distance <= threshold thành một group."""
    items = sorted(hashes.items(), key=lambda item: item[0].as_posix())
    values = np.array([value for _, value in items], dtype=np.uint64)
    count = len(items)

    parent = list(range(count))
    rank = [0] * count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    for index in range(count):
        if index + 1 >= count:
            break
        distances = np.bitwise_count(values[index] ^ values[index + 1 :])
        for offset in np.nonzero(distances <= threshold)[0]:
            union(index, index + 1 + int(offset))

    groups: dict[int, list[Path]] = defaultdict(list)
    for index, (path, _) in enumerate(items):
        groups[find(index)].append(path)

    ordered = sorted(groups.values(), key=lambda paths: paths[0].as_posix())
    mapping: dict[Path, str] = {}
    for group_index, paths in enumerate(ordered, start=1):
        group_id = f"g{group_index:06d}"
        for path in paths:
            mapping[path] = group_id
    return mapping


def validate_ratios(ratios: dict[str, float] | Iterable[float]) -> dict[str, float]:
    """Chuẩn hóa và kiểm tra tỷ lệ train/val/test."""
    if not isinstance(ratios, dict):
        ratios = dict(zip(SPLITS, ratios, strict=True))
    missing = set(SPLITS) - set(ratios)
    if missing:
        raise SplitError(f"Thiếu tỷ lệ cho: {sorted(missing)}")
    normalized = {split: float(ratios[split]) for split in SPLITS}
    if any(value <= 0 for value in normalized.values()):
        raise SplitError(f"Tỷ lệ phải > 0: {normalized}")
    if abs(sum(normalized.values()) - 1.0) > 1e-6:
        raise SplitError(f"Tổng tỷ lệ phải bằng 1: {normalized}")
    return normalized


def assign_splits(
    records: list[ImageRecord],
    group_map: dict[Path, str],
    ratios: dict[str, float],
    *,
    seed: int = 42,
) -> dict[Path, str]:
    """Chia theo group, stratified theo (dataset, label)."""
    ratios = validate_ratios(ratios)
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_group[group_map[record.path]].append(record)

    group_key: dict[str, tuple[str, str]] = {}
    for group_id, members in by_group.items():
        counts = Counter((member.dataset, member.label) for member in members)
        group_key[group_id] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, key in group_key.items():
        strata[key].append(group_id)

    rng = random.Random(seed)
    split_of_group: dict[str, str] = {}

    for key in sorted(strata):
        group_ids = sorted(strata[key])
        rng.shuffle(group_ids)
        total = sum(len(by_group[group_id]) for group_id in group_ids)
        targets = {split: total * ratios[split] for split in SPLITS}
        counts = dict.fromkeys(SPLITS, 0)

        for group_id in group_ids:
            size = len(by_group[group_id])
            best_split = SPLITS[0]
            best_score = float("-inf")
            for split in SPLITS:
                score = (targets[split] - counts[split]) / targets[split]
                if score > best_score + 1e-12:
                    best_split, best_score = split, score
            split_of_group[group_id] = best_split
            counts[best_split] += size

        # Mỗi split phải có ít nhất một group nếu stratum đủ lớn.
        if len(group_ids) >= len(SPLITS):
            for split in SPLITS:
                if counts[split] != 0:
                    continue
                donor = max(SPLITS, key=lambda name: counts[name])
                candidates = [
                    group_id
                    for group_id in group_ids
                    if split_of_group[group_id] == donor
                ]
                moved = min(candidates, key=lambda group_id: (len(by_group[group_id]), group_id))
                split_of_group[moved] = split
                counts[donor] -= len(by_group[moved])
                counts[split] += len(by_group[moved])

    return {record.path: split_of_group[group_map[record.path]] for record in records}


def validate_split(
    records: list[ImageRecord],
    split_map: dict[Path, str],
    group_map: dict[Path, str],
) -> None:
    """Đảm bảo mọi ảnh có split và không group nào nằm ở 2 split."""
    missing = [str(record.path) for record in records if record.path not in split_map]
    if missing:
        raise SplitError(f"Thiếu split cho {len(missing)} ảnh, ví dụ: {missing[:3]}")

    invalid = {split for split in split_map.values() if split not in SPLITS}
    if invalid:
        raise SplitError(f"Split không hợp lệ: {sorted(invalid)}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for path, split in split_map.items():
        group_splits[group_map[path]].add(split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        example = next(iter(leaked.items()))
        raise SplitError(f"Có group nằm ở nhiều split: {example}")


def build_rows(
    records: list[ImageRecord],
    split_map: dict[Path, str],
    group_map: dict[Path, str],
) -> list[SplitRow]:
    rows = [
        SplitRow(
            path=record.path.as_posix(),
            dataset=record.dataset,
            label=record.label,
            raw_label=record.raw_label,
            split=split_map[record.path],
            group_id=group_map[record.path],
        )
        for record in records
    ]
    return sorted(rows, key=lambda row: row.path)


def write_split_csv(rows: list[SplitRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "dataset", "label", "raw_label", "split", "group_id"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_summary(
    records: list[ImageRecord],
    group_map: dict[Path, str],
    rows: list[SplitRow],
    *,
    seed: int,
    ratios: dict[str, float],
    phash_threshold: int,
) -> dict:
    group_sizes = Counter(group_map.values())
    split_class: dict[str, dict[str, int]] = {
        split: _counts(row.label for row in rows if row.split == split) for split in SPLITS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "ratios": ratios,
        "phash_threshold": phash_threshold,
        "classes": list(CANONICAL_CLASSES),
        "counts": {
            "total": len(records),
            "by_split": _counts(row.split for row in rows),
            "by_dataset": _counts(row.dataset for row in rows),
            "by_class": _counts(row.label for row in rows),
            "by_split_class": split_class,
        },
        "groups": {
            "total": len(group_sizes),
            "duplicate_images": sum(size - 1 for size in group_sizes.values()),
            "largest": max(group_sizes.values()) if group_sizes else 0,
        },
    }


def make_splits(
    raw_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    ratios: dict[str, float] | Iterable[float] = DEFAULT_RATIOS,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    dedup: bool = True,
    force: bool = False,
) -> dict:
    """Tạo split v1 và ghi CSV + summary."""
    raw_root = Path(raw_root).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"{output_dir} đã có dữ liệu. Dùng --force hoặc tạo version mới."
        )

    ratios = validate_ratios(ratios)
    records = collect_records(raw_root)
    data_root = raw_root.parent

    if dedup:
        hashes = {record.path: compute_phash(data_root / record.path) for record in records}
        group_map = find_groups(hashes, phash_threshold)
    else:
        group_map = {
            record.path: f"g{index:06d}" for index, record in enumerate(records, start=1)
        }

    split_map = assign_splits(records, group_map, ratios, seed=seed)
    validate_split(records, split_map, group_map)
    rows = build_rows(records, split_map, group_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(rows, output_dir / "split.csv")
    summary = build_summary(
        records,
        group_map,
        rows,
        seed=seed,
        ratios=ratios,
        phash_threshold=phash_threshold,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
