"""Tạo bundle dữ liệu thô để upload lên Google Drive.

Cách dùng:
    uv run python scripts/make_data_bundle.py
    uv run python scripts/make_data_bundle.py --output dist/waste_raw_v1.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/make_data_bundle.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bundle import create_data_bundle  # noqa: E402
from src.data.download import count_images, read_manifest  # noqa: E402
from src.data.manifest import DATASETS  # noqa: E402
from src.utils.paths import raw_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đóng gói data/raw thành một file zip.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=raw_root(),
        help="Thư mục dữ liệu thô (mặc định: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/waste_raw_v1.zip"),
        help="Đường dẫn file zip đầu ra.",
    )
    return parser


def _dataset_root(raw_dir: Path, name: str) -> Path:
    manifest = read_manifest(raw_dir / name / "MANIFEST.json")
    if manifest and manifest.get("root_name"):
        candidate = raw_dir / name / str(manifest["root_name"])
        if candidate.is_dir():
            return candidate
    subdirs = [p for p in (raw_dir / name).iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"Không xác định được thư mục dữ liệu của {name}: {subdirs}")
    return subdirs[0]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)

    for name in DATASETS:
        if not (args.raw_root / name).is_dir():
            print(f"Thiếu dataset {name} trong {args.raw_root}")
            return 1
        counts = count_images(_dataset_root(args.raw_root, name))
        print(f"[{name}] {sum(counts.values())} ảnh / {len(counts)} lớp")

    info = create_data_bundle(args.raw_root, args.output)
    print(f"Đã tạo bundle: {info.path}")
    print(f"  files:  {info.files}")
    print(f"  bytes:  {info.bytes} ({info.bytes / 1024**2:.1f} MiB)")
    print(f"  sha256: {info.sha256}")
    print(f"  checksum: {info.path.with_suffix(info.path.suffix + '.sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
