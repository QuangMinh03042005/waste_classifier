"""Gom các run thực nghiệm thành một file zip (chỉ dùng trên Colab).

Local chỉ cần giữ thư mục `experiments/`; zip dùng để tải kết quả từ Colab về máy.

Cách dùng:
    uv run python scripts/export_experiments.py --rq rq1
    uv run python scripts/export_experiments.py --rq rq1 --download
    uv run python scripts/export_experiments.py --experiment rq1_customcnn --checkpoints all
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.export import create_export, download_on_colab, select_runs  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export experiments thành một file zip.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=default_experiments_root(),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rq", choices=["rq1", "rq2", "rq3"], default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--include", nargs="+", default=None, help="Danh sách run_id.")
    parser.add_argument(
        "--checkpoints",
        choices=["none", "best", "all"],
        default="best",
        help="none: bỏ checkpoint; best: chỉ best.pt; all: tất cả.",
    )
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download", action="store_true", help="Tải bằng google.colab.")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    root = args.experiments_root.resolve()
    runs = select_runs(
        root,
        rq=args.rq,
        experiment=args.experiment,
        include=args.include,
        include_incomplete=args.include_incomplete,
    )
    if not runs:
        print("Không có run nào khớp điều kiện.")
        return 1

    if args.dry_run:
        print(f"Sẽ export {len(runs)} run:")
        for run_dir in runs:
            print(f"  {run_dir.relative_to(root).as_posix()}")
        return 0

    output = args.output
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output = root / "exports" / f"waste_experiments_{stamp}.zip"

    info = create_export(
        root,
        output,
        runs,
        checkpoints=args.checkpoints,
        force=args.force,
    )
    print(f"Đã tạo: {info.path}")
    print(f"  runs:  {info.runs}")
    print(f"  files: {info.files}")
    print(f"  bytes: {info.bytes} ({info.bytes / 1024**2:.1f} MiB)")
    print(f"  sha256: {info.sha256}")

    if args.download:
        if download_on_colab(info.path):
            print("Đã gửi file qua google.colab.files.download.")
        else:
            print("Không phải Colab; tải thủ công file ở đường dẫn trên.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
