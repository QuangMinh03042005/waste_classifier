"""Đánh giá checkpoint RQ1 trên external test GD.

Chạy sau khi RQ1 final hoàn tất.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluator import evaluate_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
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


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Không có dữ liệu._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá external GD cho RQ1.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument(
        "--external-csv",
        type=Path,
        default=Path("data/splits/external/gd_test.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/RQ1/external_validation.md"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--runs", nargs="+", type=Path, default=None)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    runs = args.runs or discover_runs(args.experiments_root)
    if not runs:
        print("Chưa có run final RQ1.")
        return 1

    rows: list[dict] = []
    for run in runs:
        run_data = json.loads((run / "run.json").read_text(encoding="utf-8"))
        config = load_config(run / "config.yaml")
        metrics_file = run / "metrics.json"
        metrics = (
            json.loads(metrics_file.read_text(encoding="utf-8"))
            if metrics_file.is_file()
            else {}
        )
        checkpoint = run / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            print(f"SKIP {run.name}: không có best.pt")
            continue
        output_dir = run / "evaluations" / "external_gd"
        result = evaluate_checkpoint(
            config=config,
            checkpoint=checkpoint,
            split="test",
            datasets=["gd"],
            output_dir=output_dir,
            split_csv=args.external_csv,
            device=args.device,
        )
        in_domain = (metrics.get("test") or {}).get("macro_f1")
        rows.append(
            {
                "run_id": run_data.get("run_id"),
                "experiment": run_data.get("experiment"),
                "seed": run_data.get("seed"),
                "in_domain_macro_f1": in_domain,
                "external_macro_f1": result.metrics["macro_f1"],
                "external_accuracy": result.metrics["accuracy"],
                "drop": None if in_domain is None else in_domain - result.metrics["macro_f1"],
            }
        )
        print(
            f"{run_data.get('run_id')}: external Macro-F1 "
            f"{result.metrics['macro_f1']:.4f}"
        )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print("Không có kết quả external.")
        return 1
    summary = (
        frame.groupby("experiment")
        .agg(
            seeds=("seed", "nunique"),
            external_macro_f1_mean=("external_macro_f1", "mean"),
            external_macro_f1_std=("external_macro_f1", "std"),
            external_accuracy_mean=("external_accuracy", "mean"),
            in_domain_macro_f1_mean=("in_domain_macro_f1", "mean"),
            drop_mean=("drop", "mean"),
        )
        .reset_index()
    )
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    summary.to_csv(args.output.with_suffix(".summary.csv"), index=False)
    lines = [
        "# RQ1 – External validation trên GD",
        "",
        "External test chỉ dùng để đánh giá sau khi khóa mô hình; không dùng để chọn model.",
        "",
        "## Tổng hợp",
        "",
        _markdown(summary),
        "",
        "## Từng seed",
        "",
        _markdown(frame),
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
