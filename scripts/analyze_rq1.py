"""Phân tích kết quả RQ1: bảng, learning curves, confusion matrix, lỗi.

Cách dùng:
    uv run python scripts/analyze_rq1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

EXPERIMENTS = {
    "rq1_customcnn": "Custom CNN",
    "rq1_mobilenetv3s_full": "MobileNetV3_small pretrained",
    "rq1_mobilenetv3s_scratch": "MobileNetV3_small scratch",
}
METRICS = ["macro_f1", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_runs(
    root: Path,
    exclude: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    excluded = set(exclude or [])
    data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment in EXPERIMENTS:
        if experiment in excluded:
            continue
        for run_file in sorted((root / "rq1").glob(f"{experiment}_s*/run.json")):
            run = _load_json(run_file) or {}
            if run.get("status") != "completed":
                continue
            metrics = _load_json(run_file.with_name("metrics.json")) or {}
            history = [
                json.loads(line)
                for line in (run_file.with_name("history.jsonl")).read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            predictions_file = run_file.with_name("predictions") / "test.csv"
            predictions = (
                pd.read_csv(predictions_file) if predictions_file.is_file() else None
            )
            data[experiment].append(
                {
                    "run_dir": run_file.parent,
                    "run": run,
                    "metrics": metrics,
                    "history": history,
                    "predictions": predictions,
                }
            )
    return data


def _summary_rows(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for experiment, runs in data.items():
        for run in runs:
            test = run["metrics"].get("test") or {}
            best = run["metrics"].get("best_val") or {}
            efficiency = run["metrics"].get("efficiency") or {}
            model = run["run"].get("model") or {}
            rows.append(
                {
                    "experiment": experiment,
                    "label": EXPERIMENTS[experiment],
                    "seed": run["run"].get("seed"),
                    **{metric: test.get(metric) for metric in METRICS},
                    "best_val_macro_f1": best.get("macro_f1"),
                    "params": model.get("params"),
                    "duration_sec": run["run"].get("duration_sec"),
                    "inference_ms_per_image": efficiency.get("inference_ms_per_image"),
                }
            )
    return pd.DataFrame(rows)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    grouped = summary.groupby(["experiment", "label"], dropna=False)
    rows = []
    for (experiment, label), group in grouped:
        row: dict[str, Any] = {
            "experiment": experiment,
            "label": label,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        for metric in ("params", "duration_sec", "inference_ms_per_image"):
            row[metric] = pd.to_numeric(group[metric], errors="coerce").dropna().mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)


def _per_class(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for experiment, runs in data.items():
        for run in runs:
            test = run["metrics"].get("test") or {}
            for class_name, values in (test.get("per_class") or {}).items():
                rows.append(
                    {
                        "experiment": experiment,
                        "label": EXPERIMENTS[experiment],
                        "class": class_name,
                        **values,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["experiment", "label", "class"], dropna=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", lambda values: values.std(ddof=1)),
            recall_mean=("recall", "mean"),
        )
        .reset_index()
    )


def _confusion(data: dict[str, list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for experiment, runs in data.items():
        total: np.ndarray | None = None
        for run in runs:
            matrix = (run["metrics"].get("test") or {}).get("confusion_matrix")
            if matrix is None:
                continue
            array = np.asarray(matrix, dtype=float)
            total = array if total is None else total + array
        if total is not None:
            matrices[experiment] = total
    return matrices


def _error_tables(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for experiment, runs in data.items():
        pairs: Counter[tuple[str, str]] = Counter()
        source_errors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for run in runs:
            predictions = run["predictions"]
            if predictions is None or predictions.empty:
                continue
            wrong = predictions[predictions["correct"] == 0]
            for true_label, pred_label in zip(
                wrong["true_label"], wrong["pred_label"], strict=True
            ):
                pairs[(true_label, pred_label)] += 1
            for dataset, group in predictions.groupby("dataset"):
                source_errors[dataset][0] += int((group["correct"] == 0).sum())
                source_errors[dataset][1] += len(group)
        for (true_label, pred_label), count in pairs.most_common(5):
            rows.append(
                {
                    "experiment": experiment,
                    "label": EXPERIMENTS[experiment],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": count,
                }
            )
        for dataset, (errors, total) in sorted(source_errors.items()):
            if total:
                rows.append(
                    {
                        "experiment": experiment,
                        "label": EXPERIMENTS[experiment],
                        "true_label": f"[{dataset}]",
                        "pred_label": "error_rate",
                        "count": round(errors / total, 4),
                    }
                )
    return pd.DataFrame(rows)


def _curves(data: dict[str, list[dict[str, Any]]], figures: Path) -> list[str]:
    if not any(data.values()):
        return []
    figure, axis = plt.subplots(figsize=(8, 5))
    for experiment, runs in data.items():
        series = []
        for run in runs:
            values = [
                record["macro_f1"]
                for record in run["history"]
                if record.get("phase") == "val" and "macro_f1" in record
            ]
            if values:
                series.append(values)
        if not series:
            continue
        max_len = max(len(values) for values in series)
        matrix = np.full((len(series), max_len), np.nan)
        for index, values in enumerate(series):
            matrix[index, : len(values)] = values
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        epochs = np.arange(1, max_len + 1)
        axis.plot(epochs, mean, label=EXPERIMENTS[experiment])
        axis.fill_between(epochs, mean - std, mean + std, alpha=0.15)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Macro-F1")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = figures / "val_macro_f1_curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path.name]


def _confusion_figure(matrices: dict[str, np.ndarray], figures: Path) -> list[str]:
    if not matrices:
        return []
    figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
    if len(matrices) == 1:
        axes = [axes]
    for axis, (experiment, matrix) in zip(axes, matrices.items(), strict=True):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = matrix / np.where(row_sums == 0, 1, row_sums)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(EXPERIMENTS[experiment])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes, fraction=0.03)
    figure.tight_layout()
    path = figures / "confusion_matrices.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path.name]


def _bar_figure(aggregate: pd.DataFrame, figures: Path) -> list[str]:
    if aggregate.empty:
        return []
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(aggregate))
    axis.bar(positions, aggregate["macro_f1_mean"], yerr=aggregate["macro_f1_std"], capsize=4)
    axis.set_xticks(positions, aggregate["label"], rotation=15)
    axis.set_ylabel("Test Macro-F1")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = figures / "test_macro_f1.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path.name]


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Chưa có dữ liệu._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown(
    aggregate: pd.DataFrame,
    per_class: pd.DataFrame,
    errors: pd.DataFrame,
    figures: list[str],
) -> str:
    lines = [
        "# Kết quả RQ1",
        "",
        "Sinh tự động bởi `scripts/analyze_rq1.py`; test chỉ được đánh giá một lần.",
        "",
    ]
    if aggregate.empty:
        lines.append("_Chưa có run completed._")
        return "\n".join(lines)

    summary = aggregate[
        ["label", "n_seeds", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "params"]
    ].rename(
        columns={
            "macro_f1_mean": "Macro-F1 mean",
            "macro_f1_std": "Macro-F1 std",
            "accuracy_mean": "Accuracy mean",
            "params": "Params",
        }
    )
    lines.extend(["## Tổng hợp", "", _markdown_table(summary, list(summary.columns)), ""])

    if not per_class.empty:
        table = per_class[["label", "class", "f1_mean", "f1_std", "recall_mean"]]
        lines.extend(
            ["## Per-class F1", "", _markdown_table(table, list(table.columns)), ""]
        )

    if not errors.empty:
        table = errors[["label", "true_label", "pred_label", "count"]]
        lines.extend(
            ["## Lỗi phổ biến", "", _markdown_table(table, list(table.columns)), ""]
        )

    if figures:
        lines.append("## Hình")
        lines.append("")
        for figure in figures:
            lines.append(f"- `{figure}`")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phân tích RQ1.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument("--output-md", type=Path, default=Path("docs/RQ1/rq1_results.md"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/RQ1/rq1_analysis"))
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Loại experiment khỏi phân tích, ví dụ scratch đang chờ sửa.",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    data = load_runs(args.experiments_root.resolve(), exclude=args.exclude)
    summary = _summary_rows(data)
    aggregate = _aggregate(summary)
    per_class = _per_class(data)
    errors = _error_tables(data)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    figures.extend(_curves(data, args.figures_dir))
    figures.extend(_confusion_figure(_confusion(data), args.figures_dir))
    figures.extend(_bar_figure(aggregate, args.figures_dir))

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        build_markdown(aggregate, per_class, errors, figures) + "\n",
        encoding="utf-8",
    )
    print(f"Đã ghi {args.output_md}")
    print(f"Figures: {args.figures_dir} ({len(figures)} hình)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
