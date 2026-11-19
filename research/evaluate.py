"""Entrypoint đánh giá checkpoint (RQ3, phân tích bổ sung).

Ví dụ:
    python -m research.evaluate \
        --config configs/rq1/rq1_mobilenetv3s_full.yaml \
        --checkpoint experiments/rq1/rq1_mobilenetv3s_full_s42/checkpoints/best.pt \
        --split-csv data/splits/v1/split.csv \
        --split test --datasets realwaste \
        --output-dir experiments/rq1/rq1_mobilenetv3s_full_s42/evaluations/realwaste
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluator import evaluate_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên split/nguồn.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Lọc dataset, ví dụ realwaste, trashnet, gd.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = evaluate_checkpoint(
        config=config,
        checkpoint=args.checkpoint,
        split=args.split,
        datasets=args.datasets,
        output_dir=args.output_dir,
        split_csv=args.split_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    metrics = result.metrics
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split} | datasets: {args.datasets}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Predictions: {result.predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
