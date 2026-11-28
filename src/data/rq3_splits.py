"""Sinh split cho RQ3: cross-dataset và pooled.

Không sửa split v1. Các file mới nằm ở `data/splits/rq3/`.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.data.split import SplitError, SplitRow, read_split_csv, write_split_csv

SCHEMA_VERSION = "1.0"
MIX_TRASHNET_SHARES = (0.0, 0.25, 0.5, 0.75)


def _exclude_cross_dataset_groups(
    rows: list[SplitRow],
) -> tuple[list[SplitRow], dict[str, list[str]]]:
    group_datasets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_datasets[row.group_id].add(row.dataset)

    excluded = {
        group: sorted(datasets) for group, datasets in group_datasets.items() if len(datasets) > 1
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
            "by_split_dataset": _counts(split_rows, lambda row: f"{row.split}/{row.dataset}"),
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


def _subsample_groups(rows: list[SplitRow], *, share: float, seed: int) -> list[SplitRow]:
    """Chọn `share` số group train theo từng lớp, giữ nguyên group (chống leakage)."""
    if not 0.0 <= share <= 1.0:
        raise SplitError(f"share phải trong [0, 1], nhận {share}")

    groups_by_label: dict[str, dict[str, list[SplitRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups_by_label[row.label][row.group_id].append(row)

    rng = random.Random(seed)
    selected: list[SplitRow] = []
    for label in sorted(groups_by_label):
        groups = groups_by_label[label]
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        keep = min(len(group_ids), round(share * len(group_ids)))
        if share > 0 and keep == 0:
            keep = 1
        for group_id in group_ids[:keep]:
            selected.extend(groups[group_id])
    return selected


def make_mix_split(
    rows: list[SplitRow],
    *,
    trashnet_share: float,
    seed: int = 42,
) -> list[SplitRow]:
    """Train = toàn bộ RealWaste train + `trashnet_share` group TrashNet train.

    Val/test giữ nguyên để mọi mức tỉ lệ được đánh giá trên cùng một tập.
    """
    kept = [row for row in rows if row.split in {"val", "test"}]
    realwaste_train = [row for row in rows if row.dataset == "realwaste" and row.split == "train"]
    trashnet_train = [row for row in rows if row.dataset == "trashnet" and row.split == "train"]
    kept.extend(realwaste_train)
    kept.extend(_subsample_groups(trashnet_train, share=trashnet_share, seed=seed))

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in kept:
        group_splits[row.group_id].add(row.split)
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise SplitError(f"Group nằm ở nhiều split: {next(iter(leaked.items()))}")
    return kept


def make_mix_splits(
    source_split: Path,
    output_dir: Path,
    *,
    shares: Sequence[float] = MIX_TRASHNET_SHARES,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """Sinh các split mix-ratio cho RQ3 (RealWaste 100% + TrashNet theo tỉ lệ)."""
    source_split = Path(source_split)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        existing = sorted(output_dir.glob("mix_tn_*.csv"))
        if existing:
            raise FileExistsError(f"{existing[0]} đã tồn tại. Dùng --force.")

    rows = read_split_csv(source_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    trashnet_train_groups = len(
        {row.group_id for row in rows if row.dataset == "trashnet" and row.split == "train"}
    )
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_split": str(source_split),
        "seed": seed,
        "trashnet_train_groups": trashnet_train_groups,
        "mixes": {},
    }

    for share in shares:
        name = f"mix_tn_{round(share * 100):02d}"
        split_rows = make_mix_split(rows, trashnet_share=share, seed=seed)
        write_split_csv(split_rows, output_dir / f"{name}.csv")
        selected_groups = len(
            {
                row.group_id
                for row in split_rows
                if row.dataset == "trashnet" and row.split == "train"
            }
        )
        summary["mixes"][name] = {
            "trashnet_share": share,
            "trashnet_train_groups": selected_groups,
            "counts": _counts(split_rows, lambda row: row.split),
            "by_split_dataset": _counts(split_rows, lambda row: f"{row.split}/{row.dataset}"),
        }

    (output_dir / "mix_summary.json").write_text(
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
