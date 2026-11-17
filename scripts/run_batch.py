"""Chạy nhiều config tuần tự, tự skip run đã xong và resume run dở.

Cách dùng:
    python scripts/run_batch.py \
        --configs configs/rq1/debug/customcnn_d0.yaml configs/rq1/debug/customcnn_d1_50ep.yaml \
        --plan plans/rq1/custom_cnn_debug.yaml \
        --seeds 42

Trên Colab:
    python scripts/run_batch.py --configs ... --plan ... --seeds 42 --export
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.runner import run_training  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.export import create_export  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy batch experiment.")
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=default_experiments_root(),
    )
    parser.add_argument(
        "--clean-incomplete",
        action="store_true",
        help="Xóa run dở không có last.pt rồi chạy lại.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Tiếp tục config tiếp theo nếu một run lỗi.",
    )
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export", action="store_true", help="Export zip sau khi chạy.")
    return parser


def _read_status(run_dir: Path) -> str | None:
    run_file = run_dir / "run.json"
    if not run_file.is_file():
        return None
    try:
        return json.loads(run_file.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    root = args.experiments_root.resolve()
    failures: list[str] = []
    completed_runs: list[Path] = []

    for config_path in args.configs:
        config = load_config(config_path)
        experiment = str(config["experiment"])
        rq = str(config["rq"])

        for seed in args.seeds:
            run_id = f"{experiment}_s{seed}"
            run_dir = root / rq / run_id
            status = _read_status(run_dir)

            if status == "completed" and (run_dir / "metrics.json").is_file():
                print(f"SKIP {run_id} (completed)")
                completed_runs.append(run_dir)
                continue

            resume = False
            if run_dir.exists():
                has_last = (run_dir / "checkpoints" / "last.pt").is_file()
                if has_last:
                    print(f"RESUME {run_id} (status={status})")
                    resume = True
                elif args.clean_incomplete:
                    print(f"DELETE {run_id} (không có last.pt)")
                    shutil.rmtree(run_dir)
                else:
                    message = (
                        f"{run_id} tồn tại nhưng không có last.pt; "
                        "dùng --clean-incomplete hoặc xóa thủ công."
                    )
                    print(f"ERROR {message}")
                    failures.append(message)
                    if not args.continue_on_error:
                        return 1
                    continue

            if args.dry_run:
                action = "resume" if resume else "run"
                print(f"DRY-RUN {action} {run_id}")
                continue

            try:
                result = run_training(
                    config_path=config_path,
                    plan_path=args.plan,
                    seed=seed,
                    epochs=args.epochs,
                    limit_batches=args.limit_batches,
                    resume=resume,
                    experiments_root=root,
                )
                completed_runs.append(result.run_dir)
            except Exception as exc:  # noqa: BLE001 - batch runner cần báo mọi lỗi
                failures.append(f"{run_id}: {exc}")
                print(f"FAILED {run_id}: {exc}")
                if not args.continue_on_error:
                    return 1

    if args.export and completed_runs:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output = root / "exports" / f"waste_experiments_{stamp}.zip"
        unique = sorted({path for path in completed_runs if (path / "run.json").is_file()})
        info = create_export(root, output, unique, checkpoints="best")
        print(f"EXPORT {info.path} ({info.runs} run, {info.bytes / 1024**2:.1f} MiB)")

    if failures:
        print(f"Có {len(failures)} lỗi:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"Hoàn tất {len(completed_runs)} run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
