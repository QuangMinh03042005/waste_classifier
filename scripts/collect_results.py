"""Tổng hợp kết quả các run thành bảng CSV/Markdown cho báo cáo.

Cách dùng:
    uv run python scripts/collect_results.py
    uv run python scripts/collect_results.py --rq rq2 --include-failed
    uv run python scripts/collect_results.py --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

COLUMNS = [
    "run_id",
    "rq",
    "experiment",
    "seed",
    "status",
    "model",
    "pretrained",
    "params",
    "augmentation",
    "class_weighting",
    "image_size",
    "train_dataset",
    "test_dataset",
    "splits_version",
    "git_commit",
    "git_branch",
    "torch_version",
    "cuda_version",
    "gpu_name",
    "started_at",
    "duration_sec",
    "best_epoch",
    "best_val_macro_f1",
    "best_val_loss",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_macro_f1",
    "train_time_sec",
    "inference_ms_per_image",
]

PER_CLASS_COLUMNS = ["run_id", "experiment", "class", "precision", "recall", "f1", "support"]

SUMMARY_METRICS = [
    "best_val_macro_f1",
    "test_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "duration_sec",
    "inference_ms_per_image",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flatten(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _flag(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if isinstance(value, dict):
        return value.get("enabled", _nested(value, "method"))
    return value


def _torch_field(environment: dict[str, Any], key: str) -> Any:
    torch_env = environment.get("torch") or {}
    packages = environment.get("packages") or {}
    if key == "torch_version":
        return packages.get("torch") or torch_env.get("version")
    return torch_env.get(key)


def collect(
    root: Path,
    *,
    rq: str | None = None,
    include_failed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc mọi run và trả về (results, per_class)."""
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for run_file in sorted(root.glob("*/*/run.json")):
        run = _load_json(run_file)
        if not run:
            continue
        run_rq = run.get("rq") or run_file.parent.parent.name
        if rq and run_rq != rq:
            continue
        status = run.get("status", "unknown")
        if not include_failed and status != "completed":
            continue

        run_dir = run_file.parent
        metrics = _load_json(run_dir / "metrics.json") or {}
        environment = _load_json(run_dir / "environment.json") or {}
        config = run.get("config") or {}
        best = metrics.get("best_val") or {}
        test = metrics.get("test") or {}
        efficiency = metrics.get("efficiency") or {}
        model = run.get("model") or {}
        dataset = run.get("dataset") or {}
        git = run.get("git") or {}

        rows.append(
            {
                "run_id": run.get("run_id", run_dir.name),
                "rq": run_rq,
                "experiment": run.get("experiment", run_dir.name),
                "seed": run.get("seed"),
                "status": status,
                "model": model.get("name") or _nested(config, "model", "name"),
                "pretrained": model.get("pretrained") or _nested(config, "model", "pretrained"),
                "params": model.get("params") or efficiency.get("params"),
                "augmentation": _flatten(_flag(config, "augmentation")),
                "class_weighting": _flatten(
                    config.get("class_weighting", _flag(config, "class_weights"))
                ),
                "image_size": config.get("image_size") or _nested(config, "data", "image_size"),
                "train_dataset": config.get("train_dataset")
                or _nested(config, "dataset", "train"),
                "test_dataset": config.get("test_dataset") or _nested(config, "dataset", "test"),
                "splits_version": dataset.get("splits_version")
                or _nested(config, "dataset", "splits_version"),
                "git_commit": git.get("commit"),
                "git_branch": git.get("branch"),
                "torch_version": _torch_field(environment, "torch_version"),
                "cuda_version": _torch_field(environment, "cuda_version"),
                "gpu_name": _torch_field(environment, "device_name"),
                "started_at": run.get("started_at"),
                "duration_sec": run.get("duration_sec"),
                "best_epoch": run.get("best_epoch", best.get("epoch")),
                "best_val_macro_f1": best.get("macro_f1"),
                "best_val_loss": best.get("loss"),
                "test_accuracy": test.get("accuracy"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "test_precision_macro": test.get("precision_macro"),
                "test_recall_macro": test.get("recall_macro"),
                "test_macro_f1": test.get("macro_f1"),
                "train_time_sec": efficiency.get("train_time_sec"),
                "inference_ms_per_image": efficiency.get("inference_ms_per_image"),
            }
        )

        for class_name, values in (test.get("per_class") or {}).items():
            class_rows.append(
                {
                    "run_id": run.get("run_id", run_dir.name),
                    "experiment": run.get("experiment", run_dir.name),
                    "class": class_name,
                    **values,
                }
            )

    return (
        pd.DataFrame(rows, columns=COLUMNS),
        pd.DataFrame(class_rows, columns=PER_CLASS_COLUMNS),
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std theo experiment, chỉ tính run completed."""
    if results.empty:
        return pd.DataFrame()
    completed = results[results["status"] == "completed"]
    rows: list[dict[str, Any]] = []
    for experiment, group in completed.groupby("experiment", dropna=False):
        row: dict[str, Any] = {
            "experiment": experiment,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


def to_markdown(frame: pd.DataFrame, *, empty: str = "_Không có dữ liệu._") -> str:
    """Chuyển DataFrame thành bảng Markdown không cần thư viện ngoài."""
    if frame.empty:
        return empty
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    markdown: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "results.csv",
        "per_class": output_dir / "results_per_class.csv",
        "summary": output_dir / "results_summary.csv",
    }
    results.to_csv(paths["results"], index=False)
    per_class.to_csv(paths["per_class"], index=False)
    summary.to_csv(paths["summary"], index=False)

    if markdown:
        generated = datetime.now(UTC).isoformat()
        content = [
            "# Kết quả thực nghiệm",
            "",
            f"- Tạo lúc: `{generated}`",
            f"- Số run: {len(results)}",
            "",
            "## Tổng hợp theo experiment",
            "",
            to_markdown(summary),
            "",
            "## Chi tiết từng run",
            "",
            to_markdown(results),
            "",
        ]
        paths["markdown"] = output_dir / "results.md"
        paths["markdown"].write_text("\n".join(content), encoding="utf-8")

    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả thực nghiệm.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=default_experiments_root(),
        help="Thư mục chứa experiments (mặc định theo biến môi trường).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục xuất file (mặc định: experiments root).",
    )
    parser.add_argument("--rq", choices=["rq1", "rq2", "rq3"], default=None)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Bao gồm cả run chưa completed/failed.",
    )
    parser.add_argument("--markdown", action="store_true", help="Xuất thêm results.md.")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    results, per_class = collect(
        args.experiments_root, rq=args.rq, include_failed=args.include_failed
    )
    summary = summarize(results)
    output_dir = args.output_dir or args.experiments_root
    paths = write_outputs(
        output_dir, results, per_class, summary, markdown=args.markdown
    )

    print(f"Đã đọc {len(results)} run từ {args.experiments_root}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
