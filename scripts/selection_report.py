"""Sinh báo cáo chọn cấu hình Custom CNN.

Cách dùng:
    uv run python scripts/selection_report.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

PREFIX = "rq1_customcnn_cand_"


def load_candidates(config_dir: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "experiment" not in data:
            continue
        model = data.get("model") or {}
        training = data.get("training") or {}
        candidates[data["experiment"]] = {
            "config": path.name,
            "channels": model.get("channels"),
            "convs_per_stage": model.get("convs_per_stage"),
            "dropout": model.get("dropout"),
            "epochs": training.get("epochs"),
        }
    return candidates


def scan_runs(
    experiments_root: Path,
    prefix: str = PREFIX,
    *,
    experiments: list[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(experiments or [])
    runs: list[dict[str, Any]] = []
    for run_file in sorted((experiments_root / "rq1").glob(f"{prefix}*_s*/run.json")):
        run = json.loads(run_file.read_text(encoding="utf-8"))
        if run.get("status") != "completed":
            continue
        if wanted and run.get("experiment") not in wanted:
            continue
        metrics_file = run_file.with_name("metrics.json")
        metrics = (
            json.loads(metrics_file.read_text(encoding="utf-8"))
            if metrics_file.is_file()
            else {}
        )
        best = metrics.get("best_val") or {}
        model = run.get("model") or {}
        git = run.get("git") or {}
        runs.append(
            {
                "experiment": run.get("experiment"),
                "seed": run.get("seed"),
                "val_macro_f1": best.get("macro_f1"),
                "val_loss": best.get("loss"),
                "best_epoch": best.get("epoch"),
                "params": model.get("params"),
                "duration_sec": run.get("duration_sec"),
                "git_commit": (git.get("commit") or "")[:8],
                "run_id": run.get("run_id"),
            }
        )
    return runs


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(columns: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return lines


def build_markdown(
    candidates: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    title: str = "Chọn cấu hình Custom CNN",
    id_prefix: str = "rq1_customcnn_",
) -> str:
    def short_id(name: str) -> str:
        return name.replace(id_prefix, "")

    lines = [
        f"# {title}",
        "",
        "Bước chọn cấu hình chỉ dùng **validation**, không đánh giá test "
        "(`evaluate_test: false`).",
        "",
        "## Ứng viên",
        "",
    ]

    candidate_rows = [
        [
            short_id(experiment),
            values.get("channels"),
            values.get("convs_per_stage"),
            values.get("dropout"),
            values.get("epochs"),
            values.get("config"),
        ]
        for experiment, values in sorted(candidates.items())
    ]
    lines.extend(
        _table(
            ["ID", "Channels", "Conv/stage", "Dropout", "Epochs", "Config"],
            candidate_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Quy tắc chọn",
            "",
            "- Sàng lọc seed 42 cho cả 4 ứng viên.",
            "- Chọn top-2 theo val Macro-F1, chạy thêm seed 123 và 2024.",
            "- Chọn theo mean val Macro-F1; nếu chênh < 0.005, ưu tiên ít tham số/FLOPs hơn.",
            "- Không dùng test ở bất kỳ bước nào.",
            "",
            "## Kết quả",
            "",
        ]
    )

    if not runs:
        lines.append("_Chưa có run completed. Chạy các config trong `configs/rq1/selection/`._")
        return "\n".join(lines)

    run_rows = [
        [
            short_id(run["experiment"]),
            run["seed"],
            run["val_macro_f1"],
            run["val_loss"],
            run["best_epoch"],
            run["params"],
            run["duration_sec"],
            run["git_commit"],
        ]
        for run in runs
    ]
    lines.extend(
        _table(
            [
                "ID",
                "Seed",
                "Val Macro-F1",
                "Val Loss",
                "Best epoch",
                "Params",
                "Duration (s)",
                "Commit",
            ],
            run_rows,
        )
    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["experiment"], []).append(run)

    summary_rows: list[list[Any]] = []
    for experiment, items in sorted(grouped.items()):
        scores = [run["val_macro_f1"] for run in items if run["val_macro_f1"] is not None]
        mean = statistics.mean(scores) if scores else None
        std = statistics.stdev(scores) if len(scores) > 1 else None
        summary_rows.append(
            [
                short_id(experiment),
                len(scores),
                mean,
                std,
                items[0]["params"],
            ]
        )

    lines.extend(["", "## Tổng hợp theo ứng viên", ""])
    lines.extend(
        _table(
            ["ID", "Số seed", "Val Macro-F1 mean", "Std", "Params"],
            summary_rows,
        )
    )

    if summary_rows:
        ranked = sorted(
            summary_rows,
            key=lambda row: (
                row[2] if row[2] is not None else -1.0,
                -(row[4] or 0),
            ),
            reverse=True,
        )
        best = ranked[0]
        if (
            len(ranked) > 1
            and best[2] is not None
            and ranked[1][2] is not None
            and best[2] - ranked[1][2] < 0.005
            and (ranked[1][4] or float("inf")) < (best[4] or float("inf"))
        ):
            best = ranked[1]

        lines.extend(
            [
                "",
                "## Kết luận",
                "",
                f"- Chọn **{best[0]}** với val Macro-F1 trung bình "
                f"**{best[2]:.4f}** (std {_fmt(best[3])}), {best[4]} tham số.",
                "- Đây là kết quả chọn trên validation; test chưa được đánh giá.",
                "- Dùng cấu hình này cho RQ1 với budget cuối × 5 seed ở bước tiếp theo.",
            ]
        )
        if best[1] < 3:
            lines.append("- Lưu ý: ứng viên này mới có ít hơn 3 seed.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Báo cáo chọn cấu hình Custom CNN.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=default_experiments_root(),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/rq1/selection"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/RQ1/model_selection.md"),
    )
    parser.add_argument("--prefix", type=str, default=PREFIX)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Chỉ lấy các experiment này (dùng cho debug report).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Chọn cấu hình Custom CNN",
    )
    parser.add_argument("--id-prefix", type=str, default="rq1_customcnn_")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    candidates = load_candidates(args.config_dir)
    runs = scan_runs(
        args.experiments_root,
        args.prefix,
        experiments=args.experiments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_markdown(
            candidates,
            runs,
            title=args.title,
            id_prefix=args.id_prefix,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Đã ghi báo cáo: {args.output} ({len(runs)} run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
