"""Đo efficiency cho các run RQ1.

Cách dùng:
    uv run python scripts/measure_efficiency.py
    uv run python scripts/measure_efficiency.py --runs experiments/rq1/rq1_customcnn_s42 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.efficiency import EfficiencyResult, measure_run  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

PREFIXES = (
    "rq1_customcnn_s",
    "rq1_mobilenetv3s_full_s",
    "rq1_mobilenetv3s_scratch_s",
)


def discover_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for prefix in PREFIXES:
        runs.extend(sorted((root / "rq1").glob(f"{prefix}*/run.json")))
    return [path.parent for path in runs]


def _table(results: list[EfficiencyResult]) -> str:
    lines = [
        "| Run | Model | Params | GFLOPs | Latency B1 (ms) | Latency B32 (ms) | "
        "Images/s | VRAM (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: item.run_id):
        vram = "" if result.vram_mb is None else f"{result.vram_mb:.0f}"
        lines.append(
            f"| {result.run_id} | {result.model} | {result.params:,} | "
            f"{result.gflops:.3f} | {result.latency_ms_batch1:.2f} | "
            f"{result.latency_ms_batch32:.2f} | "
            f"{result.throughput_images_per_sec:.1f} | {vram} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đo params/FLOPs/latency.")
    parser.add_argument("--runs", nargs="+", type=Path, default=None)
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument("--output", type=Path, default=Path("docs/RQ1/rq1_efficiency.md"))
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    runs = args.runs or discover_runs(args.experiments_root)
    if not runs:
        print("Không có run nào.")
        return 1

    device = None
    if args.device:
        import torch

        device = torch.device(args.device)

    results = [measure_run(run, device=device) for run in runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = [
        "# RQ1 – Efficiency",
        "",
        "Sinh bởi `scripts/measure_efficiency.py`; FLOPs tính cho một ảnh 224×224.",
        "",
        _table(results),
        "",
    ]
    args.output.write_text("\n".join(content), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Đã ghi {args.output} ({len(results)} run)")
    for result in results:
        print(
            f"  {result.run_id}: {result.gflops:.3f} GFLOPs, "
            f"{result.latency_ms_batch1:.2f} ms/img"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
