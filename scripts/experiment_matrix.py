"""Sinh ma trận thí nghiệm Markdown từ các plan đã commit.

Cách dùng:
    uv run python scripts/experiment_matrix.py
    uv run python scripts/experiment_matrix.py --output docs/experiment_matrix.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.plan import ExperimentPlan, load_all_plans  # noqa: E402

RQ_TITLES = {
    "rq1": "RQ1 – Model Architecture",
    "rq2": "RQ2 – Data Augmentation & Class Weighting",
    "rq3": "RQ3 – Generalization",
}

COLUMNS = [
    "Experiment",
    "Loại",
    "Thay đổi",
    "Giữ cố định",
    "Metric",
    "Tiêu chí kết luận",
    "Seeds",
    "Status",
]


def _cell(value: object) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return text or "—"


def _table(rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return lines


def plan_rows(plan: ExperimentPlan) -> list[list[object]]:
    comparison = plan.raw["comparison"]
    baseline = plan.raw["baseline"]
    rows: list[list[object]] = [
        [
            baseline["experiment"],
            "baseline",
            "—",
            "—",
            comparison["metric"],
            comparison["success_criterion"],
            ", ".join(str(seed) for seed in plan.seeds),
            plan.status,
        ]
    ]
    for variant in plan.raw["variants"]:
        rows.append(
            [
                variant["experiment"],
                "variant",
                variant["change"],
                "; ".join(variant["fixed"]),
                comparison["metric"],
                comparison["success_criterion"],
                ", ".join(str(seed) for seed in plan.seeds),
                plan.status,
            ]
        )
    return rows


def build_markdown(plans: list[ExperimentPlan]) -> str:
    generated = datetime.now(UTC).isoformat()
    lines = [
        "# Ma trận thí nghiệm",
        "",
        f"- Tạo lúc: `{generated}`",
        f"- Số plan: {len(plans)}",
        f"- Số experiment: {sum(len(plan.experiments) for plan in plans)}",
        "",
    ]

    for rq in sorted(RQ_TITLES):
        group = [plan for plan in plans if plan.rq == rq]
        if not group:
            continue
        lines.extend([f"## {RQ_TITLES[rq]}", ""])
        for plan in sorted(group, key=lambda item: item.plan_id):
            data = plan.data
            lines.extend(
                [
                    f"### `{plan.plan_id}` — {plan.raw['title']}",
                    "",
                    f"- **Câu hỏi:** {plan.raw['question']}",
                    f"- **Giả thuyết:** {plan.raw['hypothesis']}",
                    f"- **Kết quả kỳ vọng:** {plan.raw['expected_result']}",
                    f"- **Nếu giả thuyết sai:** {plan.raw['failure_plan']}",
                    f"- **Metric:** {plan.raw['comparison']['metric']}",
                    f"- **Test policy:** {plan.raw['comparison']['test_policy']}",
                    f"- **Dữ liệu:** train = {', '.join(data.get('train_datasets', []))}; "
                    f"split = `{data['splits_version']}`; "
                    f"lớp = {', '.join(data['class_names'])}",
                    f"- **File plan:** `{plan.path.as_posix()}`",
                    "",
                ]
            )
            lines.extend(_table(plan_rows(plan)))
            lines.append("")

    return "\n".join(lines)


def write_matrix(plans_root: Path, output: Path) -> Path:
    plans = load_all_plans(plans_root)
    if not plans:
        raise FileNotFoundError(f"Không có plan nào trong {plans_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(plans), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sinh ma trận thí nghiệm từ plans/.")
    parser.add_argument("--plans-root", type=Path, default=Path("plans"))
    parser.add_argument("--output", type=Path, default=Path("docs/experiment_matrix.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    output = write_matrix(args.plans_root, args.output)
    print(f"Đã ghi ma trận thí nghiệm: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
