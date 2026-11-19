"""Entrypoint training cho RQ2.

Cách dùng:
    python -m research.rq2.train --config configs/rq2/rq2_mobilenetv3s_full.yaml \
        --plan plans/rq2/augmentation_ablation.yaml --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.runner import run_training  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train một thí nghiệm RQ2.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override số epoch (dùng cho smoke test).",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Giới hạn số batch mỗi epoch cho smoke test.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unplanned",
        action="store_true",
        help="Cho phép chạy không có plan (chỉ dùng debug).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    result = run_training(
        config_path=args.config,
        plan_path=args.plan,
        seed=args.seed,
        run_id=args.run_id,
        epochs=args.epochs,
        limit_batches=args.limit_batches,
        resume=args.resume,
        allow_unplanned=args.allow_unplanned,
    )

    print(f"Run: {result.run_dir}")
    print(f"Best val Macro-F1: {result.best_val.get('macro_f1'):.4f}")
    if result.test is not None:
        print(f"Test Macro-F1: {result.test.get('macro_f1'):.4f}")
    else:
        print("Test: không đánh giá (evaluate_test=false)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
