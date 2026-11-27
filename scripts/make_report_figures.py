"""Vẽ hình bổ sung cho báo cáo RQ1.

Cách dùng:
    uv run python scripts/make_report_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]
SEEDS = [42, 123, 2024, 7, 3407]

EXPERIMENTS = {
    "rq1_customcnn_p20": "Custom CNN",
    "rq1_mobilenetv3s_full_p20": "MobileNet pretrained",
    "rq1_mobilenetv3s_scratch_p20": "MobileNet scratch",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _per_class_scores(root: Path, experiment: str) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    for seed in SEEDS:
        metrics_path = root / "rq1" / f"{experiment}_s{seed}" / "metrics.json"
        if not metrics_path.is_file():
            continue
        per_class = (_load_json(metrics_path).get("test") or {}).get("per_class") or {}
        for name in CLASS_NAMES:
            values = per_class.get(name) or {}
            if values.get("f1") is not None:
                scores[name].append(float(values["f1"]))
    return scores


def plot_per_class_f1(root: Path, output: Path) -> Path:
    """So sánh per-class F1 của ba arm p20."""
    figure, axis = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(CLASS_NAMES))
    width = 0.26
    colors = ["#2563eb", "#16a34a", "#f97316"]
    for index, (experiment, label) in enumerate(EXPERIMENTS.items()):
        scores = _per_class_scores(root, experiment)
        means = [float(np.mean(scores[name])) if scores[name] else np.nan for name in CLASS_NAMES]
        stds = [
            float(np.std(scores[name], ddof=1)) if len(scores[name]) > 1 else 0.0
            for name in CLASS_NAMES
        ]
        axis.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=label,
            color=colors[index],
            edgecolor="black",
            linewidth=0.5,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_ylabel("F1")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def _val_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    path = run_dir / "history.jsonl"
    epochs: list[int] = []
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("phase") == "val":
            epochs.append(int(record["epoch"]))
            scores.append(float(record["macro_f1"]))
    return epochs, scores


def plot_scratch_patience(root: Path, output: Path) -> Path:
    """So sánh val Macro-F1 của patience 7 và 20 cho hai seed từng sụp."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, seed in zip(axes, [7, 3407], strict=True):
        old_dir = root / "rq1" / f"rq1_mobilenetv3s_scratch_s{seed}"
        new_dir = root / "rq1" / f"rq1_mobilenetv3s_scratch_p20_s{seed}"
        for run_dir, label, color, style in (
            (new_dir, "patience 20", "#2563eb", "-"),
            (old_dir, "patience 7", "#dc2626", "--"),
        ):
            epochs, scores = _val_curve(run_dir)
            axis.plot(epochs, scores, label=label, color=color, linewidth=1.6, linestyle=style)
        axis.axvline(8, color="#9ca3af", linestyle="--", linewidth=1)
        axis.text(8.3, 0.08, "epoch 8", color="#6b7280", fontsize=8)
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 0.8)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Validation Macro-F1")
    figure.suptitle("Scratch: patience 7 cắt trước khi mô hình thoát giai đoạn đầu")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Vẽ hình bổ sung cho báo cáo RQ1.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/RQ1/rq1_analysis"),
    )
    args = parser.parse_args(argv)
    root = args.experiments_root.resolve()
    output_dir = args.output_dir
    per_class = plot_per_class_f1(root, output_dir / "per_class_f1.png")
    patience = plot_scratch_patience(root, output_dir / "scratch_patience.png")
    print(f"Đã ghi {per_class}")
    print(f"Đã ghi {patience}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
