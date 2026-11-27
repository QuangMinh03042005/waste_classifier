"""So sánh Grad-CAM của ba arm RQ1 trên cùng bộ ảnh.

Cách dùng:
    uv run python scripts/gradcam_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.gradcam_compare import run_comparison  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="So sánh Grad-CAM ba arm RQ1.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/RQ1/rq1_analysis/gradcam_compare"),
    )
    parser.add_argument("--sources", nargs="+", choices=["test", "gd"], default=["test", "gd"])
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    root = args.experiments_root.resolve()
    runs = {
        "Custom CNN": root / "rq1" / f"rq1_customcnn_p20_s{args.seed}",
        "MobileNet pretrained": root / "rq1" / f"rq1_mobilenetv3s_full_p20_s{args.seed}",
        "MobileNet scratch": root / "rq1" / f"rq1_mobilenetv3s_scratch_p20_s{args.seed}",
    }
    missing = [label for label, path in runs.items() if not path.is_dir()]
    if missing:
        print(f"Thiếu run: {', '.join(missing)}")
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = run_comparison(
        runs,
        sources=tuple(args.sources),
        output_dir=args.output_dir,
        per_class=args.per_class,
        device=device,
    )
    for source, payload in results.items():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selection_path = args.output_dir / f"selection_{source}.csv"
        payload["selected"].to_csv(selection_path, index=False)
        print(f"{source}: {payload['summary']} ({len(payload['selected'])} ảnh)")
        print(f"  selection: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
