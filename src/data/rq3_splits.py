"""Sinh split cho RQ3: cross-dataset và pooled.

Không sửa split v1. Các file mới nằm ở `data/splits/rq3/`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from src.data.split import SplitError, SplitRow, read_split_csv, write_split_csv

SCHEMA_VERSION = "1.0"


def _exclude_cross_dataset_groups(rows: list[SplitRow]) -> tuple[list[SplitRow], dict[str, list[str]]]:
    group_datasets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_datasets[row.group_id].add(row.dataset)

    excluded = {
        group: sorted(datasets)
        for group, datasets in group_datasets.items()
        if len(datasets) > 1
    }
    if not excluded:
        return rows, {}
    return [row for row in rows if row.group_id not in excluded], excluded


def make_cross_dataset_split(
    rows: list[SplitRow],
    *,
    train_dataset: str,
    test_dataset: str,
) -> tuple[list[SplitRow], dict[str, list[str]]]:
    """Train/val trên một nguồn, test trên toàn bộ nguồn còn lại."""
    cleaned, excluded = _exclude_cross_dataset_groups(rows)
    result: list[SplitRow] = []

    for row in cleaned:
        if row.dataset == train_dataset and row.split in {"train", "val"}:
            result.append(row)
        elif row.dataset == test_dataset:
            result.append(
                SplitRow(
                    path=row.path,
                    dataset=row.dataset,
                    label=row.label,
                    raw_label=row.raw_label,
                    split="test",
                    group_id=row.group_id,
                )
            )

    if not result:
        raise SplitError(f"Split rỗng: train={train_dataset}, test={test_dataset}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in result:
        group_splits[row.group_id].add(row.split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise SplitError(f"Group nằm ở nhiều split: {next(iter(leaked.items()))}")
    return result, excluded


def _counts(rows: list[SplitRow], key) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))


def make_rq3_splits(source_split: Path, output_dir: Path, *, force: bool = False) -> dict:
    source_split = Path(source_split)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} đã có dữ liệu. Dùng --force.")

    rows = read_split_csv(source_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocols = {
        "rw_to_tn": ("realwaste", "trashnet"),
        "tn_to_rw": ("trashnet", "realwaste"),
    }
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_split": str(source_split),
        "protocols": {},
    }

    for name, (train_dataset, test_dataset) in protocols.items():
        split_rows, excluded = make_cross_dataset_split(
            rows, train_dataset=train_dataset, test_dataset=test_dataset
        )
        write_split_csv(split_rows, output_dir / f"{name}.csv")
        summary["protocols"][name] = {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "counts": _counts(split_rows, lambda row: row.split),
            "by_split_dataset": _counts(
                split_rows, lambda row: f"{row.split}/{row.dataset}"
            ),
            "excluded_groups": excluded,
        }

    # Pooled dùng nguyên split v1.
    write_split_csv(rows, output_dir / "pooled.csv")
    summary["protocols"]["pooled"] = {
        "counts": _counts(rows, lambda row: row.split),
        "by_split_dataset": _counts(rows, lambda row: f"{row.split}/{row.dataset}"),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Sinh split RQ3.")
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = make_rq3_splits(args.source_split, args.output_dir, force=args.force)
    for name, info in summary["protocols"].items():
        print(f"{name}: {info['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
