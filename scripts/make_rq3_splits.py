"""Sinh split RQ3 từ split v1.

Cách dùng:
    uv run python scripts/make_rq3_splits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rq3_splits import make_rq3_splits  # noqa: E402
from src.utils.paths import splits_dir  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sinh split RQ3.")
    parser.add_argument(
        "--source-split",
        type=Path,
        default=splits_dir() / "v1" / "split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=splits_dir() / "rq3",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    summary = make_rq3_splits(args.source_split, args.output_dir, force=args.force)
    print(f"Đã ghi split RQ3 tại {args.output_dir}")
    for name, info in summary["protocols"].items():
        print(f"  {name}: {info['counts']}")
        if info.get("excluded_groups"):
            print(f"    loại {len(info['excluded_groups'])} group trùng nguồn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
