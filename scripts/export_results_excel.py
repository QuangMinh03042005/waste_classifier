"""Xuất kết quả thí nghiệm ra Excel kèm hình đánh giá trực quan.

Cách dùng:
    uv run python scripts/export_results_excel.py --rq rq1
    uv run python scripts/export_results_excel.py --rq rq1 --include-failed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.drawing.image import Image as XLImage  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect_results import collect  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "history.jsonl"
    if not path.is_file():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _confusion(run_dir: Path) -> np.ndarray | None:
    metrics = _load_json(run_dir / "metrics.json") or {}
    matrix = (metrics.get("test") or {}).get("confusion_matrix")
    if not matrix:
        return None
    array = np.asarray(matrix, dtype=float)
    return array if array.ndim == 2 else None


def _per_class_f1(run_dir: Path) -> dict[str, float]:
    metrics = _load_json(run_dir / "metrics.json") or {}
    per_class = (metrics.get("test") or {}).get("per_class") or {}
    result: dict[str, float] = {}
    for name, values in per_class.items():
        if isinstance(values, dict) and values.get("f1") is not None:
            result[str(name)] = float(values["f1"])
    return result


def _normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(row_sums == 0, 1, row_sums)


def _plot_curves(axis: Any, history: pd.DataFrame, *, title: str) -> None:
    val = history[history["phase"] == "val"] if not history.empty else history
    if val.empty:
        axis.text(0.5, 0.5, "Không có history", ha="center", va="center")
        axis.set_axis_off()
        return
    epochs = val["epoch"].to_numpy()
    loss = val["loss"].to_numpy()
    f1 = val["macro_f1"].to_numpy()
    axis.plot(epochs, loss, color="#dc2626", label="Val loss", linewidth=1.5)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss", color="#dc2626")
    axis.tick_params(axis="y", labelcolor="#dc2626")
    twin = axis.twinx()
    twin.plot(epochs, f1, color="#2563eb", label="Val Macro-F1", linewidth=1.5)
    twin.set_ylabel("Macro-F1", color="#2563eb")
    twin.tick_params(axis="y", labelcolor="#2563eb")
    twin.set_ylim(0, 1.02)
    axis.set_title(title)
    axis.grid(alpha=0.25)


def _plot_confusion(axis: Any, matrix: np.ndarray, *, title: str) -> None:
    normalized = _normalize(matrix)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=7)
    axis.set_yticklabels(CLASS_NAMES, fontsize=7)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.02)


def _plot_per_class(axis: Any, values: dict[str, float], *, title: str) -> None:
    if not values:
        axis.text(0.5, 0.5, "Không có test", ha="center", va="center")
        axis.set_axis_off()
        return
    names = [name for name in CLASS_NAMES if name in values]
    scores = [values[name] for name in names]
    axis.bar(names, scores, color="#93c5fd", edgecolor="#1d4ed8")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25, labelsize=7)
    axis.grid(axis="y", alpha=0.25)


def plot_run(run_dir: Path, run_id: str, output: Path) -> Path:
    """Vẽ panel confusion + curves + per-class F1 cho một run."""
    history = _read_history(run_dir)
    matrix = _confusion(run_dir)
    per_class = _per_class_f1(run_dir)

    if matrix is None:
        figure, axis = plt.subplots(figsize=(7.2, 3.2))
        _plot_curves(axis, history, title=f"{run_id} — learning curves")
    else:
        figure, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        _plot_confusion(axes[0], matrix, title="Confusion matrix")
        _plot_curves(axes[1], history, title="Learning curves")
        _plot_per_class(axes[2], per_class, title="Per-class F1")
        figure.suptitle(run_id, fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_experiment(
    run_dirs: list[Path],
    experiment: str,
    output: Path,
) -> Path:
    """Vẽ panel tổng hợp nhiều seed của một experiment."""
    matrices = [m for d in run_dirs if (m := _confusion(d)) is not None]
    histories = [_read_history(d) for d in run_dirs]
    per_class_runs = [_per_class_f1(d) for d in run_dirs]

    figure, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    if matrices:
        average = np.mean([_normalize(m) for m in matrices], axis=0)
        axes[0].imshow(average, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title(f"Confusion trung bình ({len(matrices)} seed)")
        axes[0].set_xticks(range(len(CLASS_NAMES)))
        axes[0].set_yticks(range(len(CLASS_NAMES)))
        axes[0].set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=7)
        axes[0].set_yticklabels(CLASS_NAMES, fontsize=7)
        for i in range(average.shape[0]):
            for j in range(average.shape[1]):
                axes[0].text(j, i, f"{average[i, j]:.2f}", ha="center", va="center", fontsize=7)
    else:
        axes[0].text(0.5, 0.5, "Không có test", ha="center", va="center")
        axes[0].set_axis_off()

    val_frames = []
    for history in histories:
        if not history.empty and "phase" in history:
            val = history[history["phase"] == "val"]
            if not val.empty:
                val_frames.append(val[["epoch", "loss", "macro_f1"]])
    if val_frames:
        combined = pd.concat(val_frames, ignore_index=True)
        grouped = combined.groupby("epoch").agg(["mean", "std"])
        epochs = grouped.index.to_numpy()
        axes[1].plot(epochs, grouped[("loss", "mean")], color="#dc2626")
        axes[1].fill_between(
            epochs,
            grouped[("loss", "mean")] - grouped[("loss", "std")].fillna(0),
            grouped[("loss", "mean")] + grouped[("loss", "std")].fillna(0),
            color="#dc2626",
            alpha=0.15,
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val loss", color="#dc2626")
        twin = axes[1].twinx()
        twin.plot(epochs, grouped[("macro_f1", "mean")], color="#2563eb")
        twin.set_ylim(0, 1.02)
        twin.set_ylabel("Macro-F1", color="#2563eb")
        axes[1].set_title("Learning curves mean ± std")
    else:
        axes[1].text(0.5, 0.5, "Không có history", ha="center", va="center")
        axes[1].set_axis_off()

    merged: dict[str, list[float]] = {}
    for values in per_class_runs:
        for name, score in values.items():
            merged.setdefault(name, []).append(score)
    if merged:
        names = [name for name in CLASS_NAMES if name in merged]
        means = [float(np.mean(merged[name])) for name in names]
        stds = [
            float(np.std(merged[name], ddof=1)) if len(merged[name]) > 1 else 0 for name in names
        ]
        axes[2].bar(names, means, yerr=stds, capsize=3, color="#93c5fd", edgecolor="#1d4ed8")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Per-class F1 mean ± std")
        axes[2].tick_params(axis="x", rotation=25, labelsize=7)
        axes[2].grid(axis="y", alpha=0.25)
    else:
        axes[2].text(0.5, 0.5, "Không có test", ha="center", va="center")
        axes[2].set_axis_off()

    figure.suptitle(experiment, fontsize=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return output


def _safe_sheet_name(name: str, used: set[str]) -> str:
    clean = re.sub(r"[\\/*?:\[\]]", "-", name)[:31] or "Sheet"
    candidate = clean
    index = 2
    while candidate in used:
        suffix = f"-{index}"
        candidate = clean[: 31 - len(suffix)] + suffix
        index += 1
    used.add(candidate)
    return candidate


def _style_sheet(sheet: Any, frame: pd.DataFrame, *, numeric_decimals: int = 4) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column_index, column in enumerate(frame.columns, start=1):
        sheet.column_dimensions[get_column_letter(column_index)].width = min(
            max(10, len(str(column)) + 2), 42
        )
        for row_index, value in enumerate(frame[column].tolist(), start=2):
            cell = sheet.cell(row=row_index, column=column_index)
            if isinstance(value, float):
                cell.number_format = "0." + "0" * numeric_decimals
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def _write_frame(sheet: Any, frame: pd.DataFrame) -> None:
    if frame.empty:
        sheet["A1"] = "Không có dữ liệu"
        return
    for column_index, column in enumerate(frame.columns, start=1):
        sheet.cell(row=1, column=column_index, value=str(column))
    for row_index, row in enumerate(frame.itertuples(index=False), start=2):
        for column_index, value in enumerate(row, start=1):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = None
            else:
                value = value.item() if hasattr(value, "item") else value
            sheet.cell(row=row_index, column=column_index, value=value)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def build_workbook(
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    external: pd.DataFrame,
    efficiency: pd.DataFrame,
    *,
    experiment_images: dict[str, Path],
    run_images: dict[str, Path],
    output: Path,
) -> Path:
    """Tạo workbook Excel với bảng tổng hợp và hình nhúng."""
    merged = results.copy()
    if not external.empty:
        merged = merged.merge(
            external[["run_id", "external_macro_f1", "external_accuracy", "drop"]],
            on="run_id",
            how="left",
        )
    if not efficiency.empty:
        merged = merged.merge(
            efficiency[["run_id", "gflops", "latency_ms_batch1"]],
            on="run_id",
            how="left",
        )
    merged["run_image"] = merged["run_id"].map(lambda key: str(run_images.get(key, "")))
    merged["experiment_image"] = merged["experiment"].map(
        lambda key: str(experiment_images.get(key, ""))
    )
    for column in ("external_macro_f1", "external_accuracy", "drop", "gflops", "latency_ms_batch1"):
        if column not in merged.columns:
            merged[column] = np.nan

    workbook = Workbook()
    readme = workbook.active
    readme.title = "README"
    readme["A1"] = "Tổng hợp kết quả thí nghiệm"
    readme["A1"].font = Font(bold=True, size=14)
    lines = [
        ("Nguồn", "experiments/<rq>/<run>/run.json, metrics.json, history.jsonl"),
        ("Thời điểm xuất", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
        ("Số run", len(results)),
        ("Số experiment", results["experiment"].nunique() if not results.empty else 0),
        (
            "Ghi chú",
            "Test của RQ1 đã được tái sử dụng sau khi điều chỉnh protocol; "
            "không dùng cột test để chọn model.",
        ),
    ]
    for row, (key, value) in enumerate(lines, start=3):
        readme.cell(row=row, column=1, value=key).font = Font(bold=True)
        readme.cell(row=row, column=2, value=str(value))
    readme.column_dimensions["A"].width = 24
    readme.column_dimensions["B"].width = 110
    for row in readme.iter_rows(min_row=3, max_row=7, min_col=1, max_col=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    runs_sheet = workbook.create_sheet("Runs")
    run_columns = [
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
        "best_epoch",
        "best_val_macro_f1",
        "test_macro_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "external_macro_f1",
        "external_accuracy",
        "drop",
        "gflops",
        "latency_ms_batch1",
        "inference_ms_per_image",
        "duration_sec",
        "git_commit",
    ]
    run_columns = [column for column in run_columns if column in merged.columns]
    runs_frame = merged[run_columns].sort_values(["experiment", "seed"])
    _write_frame(runs_sheet, runs_frame)
    _style_sheet(runs_sheet, runs_frame)

    summary = (
        merged.groupby("experiment", dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            seeds=("seed", lambda values: ", ".join(str(v) for v in sorted(set(values)))),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", lambda values: values.std(ddof=1)),
            test_accuracy_mean=("test_accuracy", "mean"),
            external_macro_f1_mean=("external_macro_f1", "mean"),
            drop_mean=("drop", "mean"),
            gflops=("gflops", "mean"),
            latency_ms=("latency_ms_batch1", "mean"),
        )
        .reset_index()
    )
    summary_sheet = workbook.create_sheet("Summary")
    _write_frame(summary_sheet, summary)
    _style_sheet(summary_sheet, summary)

    if not per_class.empty:
        per_class_sheet = workbook.create_sheet("Per-class")
        _write_frame(per_class_sheet, per_class)
        _style_sheet(per_class_sheet, per_class)
    else:
        per_class_sheet = workbook.create_sheet("Per-class")
        per_class_sheet["A1"] = "Không có dữ liệu"

    experiments_sheet = workbook.create_sheet("Experiments")
    experiment_frame = summary.copy()
    experiment_frame["image"] = ""
    _write_frame(experiments_sheet, experiment_frame)
    _style_sheet(experiments_sheet, experiment_frame)

    image_column = get_column_letter(list(experiment_frame.columns).index("image") + 1)
    experiments_sheet.column_dimensions[image_column].width = 48
    for row_index, experiment in enumerate(experiment_frame["experiment"], start=2):
        experiment_path = experiment_images.get(str(experiment))
        if experiment_path is None or not Path(experiment_path).is_file():
            continue
        image = XLImage(str(experiment_path))
        original_width, original_height = image.width, image.height
        image.width = 360
        image.height = int(360 * original_height / original_width)
        experiments_sheet.add_image(image, f"{image_column}{row_index}")
        experiments_sheet.row_dimensions[row_index].height = max(60.0, image.height * 0.75 + 8)

    figures_sheet = workbook.create_sheet("Figures")
    figure_rows = []
    for experiment, path in sorted(experiment_images.items()):
        figure_rows.append(
            {"experiment": experiment, "run_id": "", "kind": "experiment", "path": str(path)}
        )
    for run_id, path in sorted(run_images.items()):
        figure_rows.append({"experiment": "", "run_id": run_id, "kind": "run", "path": str(path)})
    figures_frame = pd.DataFrame(figure_rows)
    _write_frame(figures_sheet, figures_frame)
    _style_sheet(figures_sheet, figures_frame)

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xuất kết quả thí nghiệm ra Excel.")
    parser.add_argument("--rq", choices=["rq1", "rq2", "rq3"], default=None)
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument("--output", type=Path, default=Path("report/rq1_results.xlsx"))
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("report/figures/excel"),
    )
    parser.add_argument("--include-failed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    root = args.experiments_root.resolve()
    results, per_class = collect(root, rq=args.rq, include_failed=args.include_failed)
    if results.empty:
        print("Không có run completed.")
        return 1

    # Chỉ giữ run thực sự nằm trong thư mục experiments/<rq>/<run_id>.
    keep = results.apply(lambda row: (root / str(row["rq"]) / str(row["run_id"])).is_dir(), axis=1)
    results = results[keep].drop_duplicates(subset=["run_id"], keep="first").reset_index(drop=True)
    per_class = per_class[per_class["run_id"].isin(results["run_id"])].reset_index(drop=True)

    external_path = Path("docs/RQ1/external_validation.csv")
    external = pd.read_csv(external_path) if external_path.is_file() else pd.DataFrame()
    if not external.empty:
        external = external[["run_id", "external_macro_f1", "external_accuracy", "drop"]]

    efficiency_path = Path("docs/RQ1/rq1_efficiency.json")
    if efficiency_path.is_file():
        efficiency = pd.DataFrame(json.loads(efficiency_path.read_text(encoding="utf-8")))
        efficiency = efficiency[["run_id", "gflops", "latency_ms_batch1"]]
    else:
        efficiency = pd.DataFrame()

    run_images: dict[str, Path] = {}
    experiment_images: dict[str, Path] = {}
    for row in results.itertuples(index=False):
        run_dir = root / str(row.rq) / str(row.run_id)
        path = args.figures_dir / "runs" / f"{row.run_id}.png"
        plot_run(run_dir, str(row.run_id), path)
        run_images[str(row.run_id)] = path

    for experiment, group in results.groupby("experiment"):
        run_dirs = [root / str(row.rq) / str(row.run_id) for row in group.itertuples(index=False)]
        path = args.figures_dir / "experiments" / f"{experiment}.png"
        plot_experiment(run_dirs, str(experiment), path)
        experiment_images[str(experiment)] = path

    output = build_workbook(
        results,
        per_class,
        external,
        efficiency,
        experiment_images=experiment_images,
        run_images=run_images,
        output=args.output,
    )
    print(f"Đã ghi {output} ({len(results)} run, {len(experiment_images)} experiment)")
    print(f"Hình: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
