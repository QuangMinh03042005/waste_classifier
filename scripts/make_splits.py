"""Tạo split train/val/test cho 5 lớp chung.

Cách dùng:
    uv run python scripts/make_splits.py
    uv run python scripts/make_splits.py --output data/splits/v1 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.split import DEFAULT_PHASH_THRESHOLD, SPLITS, make_splits  # noqa: E402
from src.utils.paths import raw_root, splits_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tạo split train/val/test.")
    parser.add_argument("--raw-root", type=Path, default=raw_root())
    parser.add_argument("--output", type=Path, default=splits_dir() / "v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="Hamming distance tối đa để coi là ảnh trùng.",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Tắt kiểm tra ảnh trùng.")
    parser.add_argument("--force", action="store_true", help="Ghi đè thư mục split đã có.")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    ratios = {"train": args.train, "val": args.val, "test": args.test}

    summary = make_splits(
        args.raw_root,
        args.output,
        seed=args.seed,
        ratios=ratios,
        phash_threshold=args.phash_threshold,
        dedup=not args.no_dedup,
        force=args.force,
    )

    counts = summary["counts"]
    print(f"Đã tạo split tại {args.output}")
    print(f"  tổng ảnh: {counts['total']}")
    for split in SPLITS:
        print(f"  {split:5s}: {counts['by_split'].get(split, 0)}")
    print(
        f"  group: {summary['groups']['total']}, "
        f"ảnh trùng: {summary['groups']['duplicate_images']}"
    )
    print("  theo lớp:")
    for label, count in counts["by_class"].items():
        print(f"    {label:10s}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
