"""Sinh thống kê và hình EDA cho bộ dữ liệu 5 lớp.

Cách dùng:
    uv run python scripts/run_eda.py
    uv run python scripts/run_eda.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.eda import (  # noqa: E402
    build_summary,
    collect_image_stats,
    duplicate_table,
    generate_all,
    load_records,
    save_tables,
)
from src.utils.paths import data_root, splits_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA cho dữ liệu 5 lớp.")
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=splits_dir() / "v1" / "split.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/EDA"))
    parser.add_argument("--force", action="store_true", help="Tính lại toàn bộ ảnh.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N ảnh (test).")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    cache = args.output_dir / "tables" / "image_stats.csv"
    if cache.is_file() and not args.force and args.limit is None:
        print(f"Dùng cache {cache}")
        stats = pd.read_csv(cache)
    else:
        records = load_records(args.split_csv)
        stats = collect_image_stats(records, args.data_root, limit=args.limit)

    paths = save_tables(stats, args.output_dir, data_root=args.data_root)
    figures = generate_all(stats, args.data_root, args.output_dir)

    summary = build_summary(stats, duplicate_table(stats))
    print(f"Đã ghi EDA tại {args.output_dir}")
    print(f"  ảnh: {summary['total']}")
    print(f"  split: {summary['by_split']}")
    print(f"  dataset: {summary['by_dataset']}")
    print(f"  imbalance ratio: {summary['imbalance_ratio']:.2f}")
    print(
        f"  duplicate: {summary['duplicate_images']} ảnh / "
        f"{summary['duplicate_groups']} group"
    )
    print(f"  quality issues: {summary['quality_issues']}")
    print(f"  tables: {len(paths)} file, figures: {len(figures)} hình")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
