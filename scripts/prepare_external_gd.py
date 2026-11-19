"""Tải và chuẩn bị external test GD cho RQ1.

Cách dùng:
    python scripts/prepare_external_gd.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.external import prepare_external_gd  # noqa: E402
from src.utils.paths import data_root, splits_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chuẩn bị external test GD.")
    parser.add_argument("--data-root", type=Path, default=data_root())
    parser.add_argument(
        "--train-split",
        type=Path,
        default=splits_dir() / "v1" / "split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=splits_dir() / "external",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    summary = prepare_external_gd(
        data_root=args.data_root,
        train_split=args.train_split,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"Đã tạo external test GD tại {args.output_dir}")
    print(f"  ảnh gốc: {summary['original_records']}")
    print(f"  loại trùng train: {summary['excluded_train_overlap']}")
    print(f"  loại trùng nội bộ: {summary['excluded_internal_duplicates']}")
    print(f"  giữ lại: {summary['kept']}")
    print(f"  theo lớp: {summary['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
