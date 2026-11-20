"""Sinh split mix-ratio cho RQ3 (RealWaste 100% + TrashNet theo tỉ lệ).

Không sửa split v1 và các split RQ3 đã có; chỉ thêm `mix_tn_*.csv` và
`mix_summary.json` trong `data/splits/rq3/`.

Cách dùng:
    uv run python scripts/make_rq3_mix_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rq3_splits import MIX_TRASHNET_SHARES, make_mix_splits  # noqa: E402
from src.utils.paths import splits_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Sinh split mix-ratio cho RQ3.")
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
    parser.add_argument("--shares", nargs="+", type=float, default=list(MIX_TRASHNET_SHARES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    summary = make_mix_splits(
        args.source_split,
        args.output_dir,
        shares=args.shares,
        seed=args.seed,
        force=args.force,
    )
    for name, info in summary["mixes"].items():
        print(
            f"{name}: share={info['trashnet_share']:.2f} "
            f"trashnet_train_groups={info['trashnet_train_groups']} "
            f"counts={info['counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
