"""Sinh bảng theo dõi thí nghiệm RQ3 ra Excel từ artifact của các run.

Cách dùng:
    uv run python scripts/track_rq3.py

Đọc `run.json`, `metrics.json`, `environment.json`, `predictions/test.csv` và các file
plan trong `plans/rq3/`; ghi `docs/RQ3/rq3_tracking.xlsx` gồm 4 sheet:
Runs, Tổng hợp, Kế hoạch, Môi trường.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

SOURCES = ("realwaste", "trashnet")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(bold=True, color="FFFFFF")
STATUS_FILL = {
    "completed": PatternFill("solid", fgColor="C6EFCE"),
    "running": PatternFill("solid", fgColor="FFEB9C"),
    "failed": PatternFill("solid", fgColor="FFC7CE"),
}

RUN_COLUMNS = [
    "ID",
    "Nhóm",
    "Experiment",
    "Seed",
    "Trạng thái",
    "Bắt đầu",
    "Thời gian (s)",
    "Commit",
    "GPU",
    "Torch",
    "Split",
    "Train",
    "Val",
    "Test",
    "Best epoch",
    "Val Macro-F1",
    "Test Macro-F1",
    "Test Accuracy",
    "Test F1 RealWaste",
    "Test F1 TrashNet",
    "Params",
    "ms/ảnh",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _group_of(experiment: str) -> str:
    if "mix_" in experiment:
        return "Mix-ratio"
    if "pooled" in experiment:
        return "Pooled"
    return "Cross-dataset"


def _source_f1(predictions: pd.DataFrame | None, source: str) -> float | None:
    if predictions is None or predictions.empty:
        return None
    subset = predictions[predictions["dataset"] == source]
    if subset.empty:
        return None
    return float(f1_score(subset["true_label"], subset["pred_label"], average="macro"))


def collect_runs(root: Path) -> list[dict[str, Any]]:
    """Đọc mọi run của RQ3 (kể cả run dở/lỗi trong experiments root)."""
    rows: list[dict[str, Any]] = []
    for run_file in sorted((root / "rq3").glob("*/run.json")):
        run = _load_json(run_file)
        if not run:
            continue
        run_dir = run_file.parent
        metrics = _load_json(run_dir / "metrics.json")
        environment = _load_json(run_dir / "environment.json")
        predictions_file = run_dir / "predictions" / "test.csv"
        predictions = pd.read_csv(predictions_file) if predictions_file.is_file() else None

        torch_info = environment.get("torch") or {}
        dataset = run.get("dataset") or {}
        efficiency = metrics.get("efficiency") or {}
        test = metrics.get("test") or {}
        best_val = metrics.get("best_val") or {}
        model = run.get("model") or {}

        rows.append(
            {
                "ID": run.get("run_id", run_dir.name),
                "Nhóm": _group_of(str(run.get("experiment", ""))),
                "Experiment": run.get("experiment"),
                "Seed": run.get("seed"),
                "Trạng thái": run.get("status"),
                "Bắt đầu": (run.get("started_at") or "")[:19].replace("T", " "),
                "Thời gian (s)": round(float(run.get("duration_sec") or 0.0), 1),
                "Commit": (run.get("git") or {}).get("commit", "")[:8],
                "GPU": torch_info.get("device_name"),
                "Torch": torch_info.get("version"),
                "Split": Path(str(dataset.get("split_csv") or "")).name,
                "Train": dataset.get("train"),
                "Val": dataset.get("val"),
                "Test": dataset.get("test"),
                "Best epoch": best_val.get("epoch") or run.get("best_epoch"),
                "Val Macro-F1": best_val.get("macro_f1"),
                "Test Macro-F1": test.get("macro_f1"),
                "Test Accuracy": test.get("accuracy"),
                "Test F1 RealWaste": _source_f1(predictions, "realwaste"),
                "Test F1 TrashNet": _source_f1(predictions, "trashnet"),
                "Params": model.get("params"),
                "ms/ảnh": efficiency.get("inference_ms_per_image"),
                "_run_id": run.get("run_id", run_dir.name),
                "_environment": environment,
                "_plan": (run.get("plan") or {}).get("plan_id"),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean ± std theo experiment trên các run completed."""
    frame = pd.DataFrame([row for row in rows if row["Trạng thái"] == "completed"])
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for (group, experiment), part in frame.groupby(["Nhóm", "Experiment"], sort=True):
        record: dict[str, Any] = {
            "Nhóm": group,
            "Experiment": experiment,
            "n_seeds": int(part["Seed"].nunique()),
            "Train": int(part["Train"].iloc[0]),
        }
        for column, plain in (
            ("Val Macro-F1", "Val Macro-F1"),
            ("Test Macro-F1", "Test Macro-F1"),
            ("Test Accuracy", "Test Accuracy"),
            ("Test F1 RealWaste", "Test F1 RealWaste"),
            ("Test F1 TrashNet", "Test F1 TrashNet"),
        ):
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            record[f"{plain} mean"] = float(values.mean()) if not values.empty else None
            record[f"{plain} std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0 if values.size else None
            )
        record["Thời gian (s) mean"] = float(
            pd.to_numeric(part["Thời gian (s)"], errors="coerce").mean()
        )
        records.append(record)
    return records


def plan_rows(plan_paths: list[Path]) -> list[dict[str, Any]]:
    """Bảng kế hoạch: baseline + từng variant của các plan RQ3."""
    records: list[dict[str, Any]] = []
    for path in plan_paths:
        if not path.is_file():
            continue
        plan = yaml.safe_load(path.read_text(encoding="utf-8"))
        budget = plan.get("budget") or {}
        base = plan.get("baseline") or {}
        records.append(
            {
                "Plan": plan.get("plan_id"),
                "Vai trò": "baseline",
                "Experiment": base.get("experiment"),
                "Thay đổi": base.get("description"),
                "Biến cố định": "",
                "Seeds": ", ".join(str(seed) for seed in plan.get("seeds", [])),
                "Max epochs": budget.get("max_epochs"),
                "Batch": budget.get("batch_size"),
                "Trạng thái plan": plan.get("status"),
                "File": str(path),
            }
        )
        for variant in plan.get("variants", []):
            records.append(
                {
                    "Plan": plan.get("plan_id"),
                    "Vai trò": "variant",
                    "Experiment": variant.get("experiment"),
                    "Thay đổi": variant.get("change"),
                    "Biến cố định": "; ".join(variant.get("fixed", [])),
                    "Seeds": ", ".join(str(seed) for seed in plan.get("seeds", [])),
                    "Max epochs": budget.get("max_epochs"),
                    "Batch": budget.get("batch_size"),
                    "Trạng thái plan": plan.get("status"),
                    "File": str(path),
                }
            )
    return records


def _write_sheet(
    workbook: Workbook,
    title: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    *,
    float_columns: set[str] | None = None,
) -> None:
    sheet = workbook.create_sheet(title)
    sheet.append(columns)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.freeze_panes = "A2"

    for row in rows:
        sheet.append([row.get(column) for column in columns])

    float_columns = float_columns or set()
    for index, column in enumerate(columns, start=1):
        values = [str(row.get(column) or "") for row in rows]
        width = max([len(column), *(len(value) for value in values)]) + 2
        sheet.column_dimensions[get_column_letter(index)].width = min(width, 60)
        if column in float_columns:
            for row_index in range(2, len(rows) + 2):
                sheet.cell(row=row_index, column=index).number_format = "0.0000"

    if "Trạng thái" in columns:
        status_index = columns.index("Trạng thái") + 1
        for row_index in range(2, len(rows) + 2):
            cell = sheet.cell(row=row_index, column=status_index)
            if cell.value in STATUS_FILL:
                cell.fill = STATUS_FILL[cell.value]


def write_workbook(
    runs: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Ghi file .xlsx với 4 sheet; trả về đường dẫn."""
    workbook = Workbook()
    workbook.remove(workbook.active)

    _write_sheet(workbook, "Runs", RUN_COLUMNS, runs, float_columns=set(RUN_COLUMNS[15:20]))

    summary_columns = ["Nhóm", "Experiment", "n_seeds", "Train"]
    for metric in (
        "Val Macro-F1",
        "Test Macro-F1",
        "Test Accuracy",
        "Test F1 RealWaste",
        "Test F1 TrashNet",
    ):
        summary_columns.extend([f"{metric} mean", f"{metric} std"])
    summary_columns.append("Thời gian (s) mean")
    _write_sheet(
        workbook,
        "Tổng hợp",
        summary_columns,
        summary,
        float_columns={column for column in summary_columns if column.endswith(("mean", "std"))},
    )

    plan_columns = [
        "Plan",
        "Vai trò",
        "Experiment",
        "Thay đổi",
        "Biến cố định",
        "Seeds",
        "Max epochs",
        "Batch",
        "Trạng thái plan",
        "File",
    ]
    _write_sheet(workbook, "Kế hoạch", plan_columns, plans)

    environment_rows = []
    for row in runs:
        environment = row.get("_environment") or {}
        torch_info = environment.get("torch") or {}
        environment_rows.append(
            {
                "Run": row["_run_id"],
                "Plan": row.get("_plan"),
                "Device": torch_info.get("device_name"),
                "Capability": torch_info.get("device_capability"),
                "VRAM (GB)": torch_info.get("vram_gb"),
                "Torch": torch_info.get("version"),
                "CUDA": torch_info.get("cuda_version"),
                "cuDNN": torch_info.get("cudnn_version"),
                "Python": environment.get("python_version"),
                "Platform": environment.get("platform"),
                "Commit": (environment.get("git") or {}).get("commit", "")[:8],
                "Branch": (environment.get("git") or {}).get("branch"),
                "Dirty": (environment.get("git") or {}).get("dirty"),
            }
        )
    environment_columns = list(environment_rows[0]) if environment_rows else ["Run"]
    _write_sheet(workbook, "Môi trường", environment_columns, environment_rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sinh tracking Excel cho RQ3.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument(
        "--plans",
        nargs="+",
        type=Path,
        default=None,
        help="Mặc định lấy mọi file trong plans/rq3/.",
    )
    parser.add_argument("--output", type=Path, default=Path("docs/RQ3/rq3_tracking.xlsx"))
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    plan_paths = args.plans or sorted((Path("plans") / "rq3").glob("*.yaml"))
    runs = collect_runs(args.experiments_root.resolve())
    summary = summarize(runs)
    plans = plan_rows(plan_paths)

    path = write_workbook(runs, summary, plans, args.output)
    completed = sum(1 for row in runs if row["Trạng thái"] == "completed")
    print(f"Đã ghi {path}")
    print(f"Runs: {len(runs)} (completed {completed}) | Tổng hợp: {len(summary)} dòng")
    print(f"Kế hoạch: {len(plans)} dòng | Plan: {len(plan_paths)} file")
    seen: dict[str, int] = defaultdict(int)
    for row in runs:
        if row["Trạng thái"] == "completed":
            seen[row["Experiment"]] += 1
    short = [f"{name}({count})" for name, count in sorted(seen.items()) if count < 3]
    if short:
        print("Chưa đủ 3 seed cho: " + ", ".join(short))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
