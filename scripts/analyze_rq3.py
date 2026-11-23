"""Phân tích kết quả RQ3: cross-dataset, pooled, domain shift.

Cách dùng:
    uv run python scripts/analyze_rq3.py

Sinh `docs/RQ3/rq3_results.md`, `docs/RQ3/rq3_metrics.csv` và hình trong
`docs/RQ3/rq3_analysis/`. Chỉ đọc artifact đã có; test không được đánh giá lại.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import CANONICAL_CLASSES  # noqa: E402
from src.utils.paths import experiments_root as default_experiments_root  # noqa: E402
from src.utils.paths import splits_dir  # noqa: E402

PROTOCOLS = {
    "rq3_train_realwaste_test_trashnet": {
        "label": "RealWaste → TrashNet",
        "train": "realwaste",
        "tests": ["trashnet"],
    },
    "rq3_train_trashnet_test_realwaste": {
        "label": "TrashNet → RealWaste",
        "train": "trashnet",
        "tests": ["realwaste"],
    },
    "rq3_train_pooled_test_each": {
        "label": "Pooled (TrashNet 100%)",
        "train": "pooled",
        "tests": ["realwaste", "trashnet"],
    },
    "rq3_mix_tn00": {
        "label": "Mix: TrashNet 0%",
        "train": "realwaste",
        "tests": ["realwaste", "trashnet"],
    },
    "rq3_mix_tn25": {
        "label": "Mix: TrashNet 25%",
        "train": "realwaste+trashnet",
        "tests": ["realwaste", "trashnet"],
    },
    "rq3_mix_tn50": {
        "label": "Mix: TrashNet 50%",
        "train": "realwaste+trashnet",
        "tests": ["realwaste", "trashnet"],
    },
    "rq3_mix_tn75": {
        "label": "Mix: TrashNet 75%",
        "train": "realwaste+trashnet",
        "tests": ["realwaste", "trashnet"],
    },
}
SINGLE_SOURCE = {
    "realwaste": "rq3_train_trashnet_test_realwaste",
    "trashnet": "rq3_train_realwaste_test_trashnet",
}
MIX_EXPERIMENTS = {
    "rq3_mix_tn00": 0,
    "rq3_mix_tn25": 25,
    "rq3_mix_tn50": 50,
    "rq3_mix_tn75": 75,
    "rq3_train_pooled_test_each": 100,
}
SOURCES = ("realwaste", "trashnet")
SUCCESS_DELTA = 0.02
T_CRITICAL = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
}


def ci95(std: float | None, n: int) -> float | None:
    """Nửa độ rộng CI 95% (phân phối t, hai phía); None nếu không đủ mẫu."""
    if std is None or n not in T_CRITICAL:
        return None
    return T_CRITICAL[n] * float(std) / math.sqrt(n)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Đọc mọi run completed của RQ3, kèm predictions nếu có."""
    data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for experiment in PROTOCOLS:
        for run_file in sorted((root / "rq3").glob(f"{experiment}_s*/run.json")):
            run = _load_json(run_file) or {}
            if run.get("status") != "completed":
                continue
            metrics = _load_json(run_file.with_name("metrics.json")) or {}
            predictions_file = run_file.with_name("predictions") / "test.csv"
            predictions = pd.read_csv(predictions_file) if predictions_file.is_file() else None
            data[experiment].append(
                {
                    "run_dir": run_file.parent,
                    "run": run,
                    "metrics": metrics,
                    "predictions": predictions,
                }
            )
    return data


def compute_metrics(
    predictions: pd.DataFrame, classes: tuple[str, ...] = CANONICAL_CLASSES
) -> dict:
    """Metric trên một tập predictions đã lọc."""
    labels = list(classes)
    y_true = predictions["true_label"]
    y_pred = predictions["pred_label"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        str(name): {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(labels)
    }
    return {
        "n": int(len(predictions)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "pred_counts": dict(Counter(y_pred)),
    }


def heldout_test_paths(split_csv: Path) -> dict[str, set[str]]:
    """Đường dẫn ảnh thuộc held-out test của split v1, theo từng nguồn."""
    if not Path(split_csv).is_file():
        return {}
    frame = pd.read_csv(split_csv)
    return {
        str(dataset): set(group["path"])
        for dataset, group in frame[frame["split"] == "test"].groupby("dataset")
    }


def _run_rows(
    data: dict[str, list[dict[str, Any]]],
    heldout: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment, runs in data.items():
        info = PROTOCOLS[experiment]
        for run in runs:
            predictions = run["predictions"]
            if predictions is None or predictions.empty:
                continue
            best = (run["metrics"].get("best_val") or {}).get("macro_f1")
            model = run["run"].get("model") or {}
            for source in info["tests"]:
                subset = predictions[predictions["dataset"] == source]
                if subset.empty:
                    continue
                record = {
                    "experiment": experiment,
                    "label": info["label"],
                    "train": info["train"],
                    "test_source": source,
                    "seed": run["run"].get("seed"),
                    "best_val_macro_f1": best,
                    "params": model.get("params"),
                    **compute_metrics(subset),
                }
                fair = _fair_subset(subset, source, heldout)
                if fair is not None:
                    fair_metrics = compute_metrics(fair)
                    record["heldout_macro_f1"] = fair_metrics["macro_f1"]
                    record["heldout_accuracy"] = fair_metrics["accuracy"]
                    record["n_heldout"] = fair_metrics["n"]
                rows.append(record)
            if len(info["tests"]) > 1:
                rows.append(
                    {
                        "experiment": experiment,
                        "label": info["label"],
                        "train": info["train"],
                        "test_source": "all",
                        "seed": run["run"].get("seed"),
                        "best_val_macro_f1": best,
                        "params": model.get("params"),
                        **compute_metrics(predictions),
                    }
                )
    return pd.DataFrame(rows)


def _fair_subset(
    subset: pd.DataFrame,
    source: str,
    heldout: dict[str, set[str]] | None,
) -> pd.DataFrame | None:
    """Ảnh thuộc held-out test của nguồn; None nếu tập hiện tại đã đúng held-out."""
    if not heldout or source not in heldout:
        return None
    paths = heldout[source]
    fair = subset[subset["path"].isin(paths)]
    if fair.empty or len(fair) == len(subset):
        return None
    return fair


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std trên các seed, theo protocol và nguồn test."""
    if rows.empty:
        return rows
    grouped = rows.groupby(["experiment", "label", "train", "test_source"], dropna=False)
    records: list[dict[str, Any]] = []
    for (experiment, label, train, source), group in grouped:
        record: dict[str, Any] = {
            "experiment": experiment,
            "label": label,
            "train": train,
            "test_source": source,
            "n_seeds": int(group["seed"].nunique()),
            "n_test": int(group["n"].iloc[0]),
            "best_val_macro_f1": float(pd.to_numeric(group["best_val_macro_f1"]).mean()),
            "params": int(pd.to_numeric(group["params"]).dropna().iloc[0])
            if group["params"].notna().any()
            else None,
        }
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_ci95"] = ci95(record[f"{metric}_std"], len(values))
        if "heldout_macro_f1" in group and group["heldout_macro_f1"].notna().any():
            values = pd.to_numeric(group["heldout_macro_f1"], errors="coerce").dropna()
            record["heldout_macro_f1_mean"] = float(values.mean())
            record["heldout_macro_f1_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record["heldout_macro_f1_ci95"] = ci95(record["heldout_macro_f1_std"], len(values))
            record["n_heldout"] = int(group["n_heldout"].dropna().iloc[0])
        records.append(record)
    order = {"realwaste": 0, "trashnet": 1, "all": 2}
    frame = pd.DataFrame(records)
    frame["_order"] = frame["test_source"].map(order)
    frame = frame.sort_values(["_order", "macro_f1_mean"], ascending=[True, False])
    return frame.drop(columns="_order")


def compare_pooled(agg: pd.DataFrame) -> pd.DataFrame:
    """So sánh pooled (100%) với single-source trên từng nguồn test."""
    if agg.empty:
        return agg
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        single = agg[(agg["test_source"] == source) & (agg["experiment"] == SINGLE_SOURCE[source])]
        pooled = agg[
            (agg["test_source"] == source) & (agg["experiment"] == "rq3_train_pooled_test_each")
        ]
        if single.empty or pooled.empty:
            continue
        single_row = single.iloc[0]
        pooled_row = pooled.iloc[0]
        delta = float(pooled_row["macro_f1_mean"]) - float(single_row["macro_f1_mean"])
        record = {
            "test_source": source,
            "single_label": single_row["label"],
            "single_macro_f1": float(single_row["macro_f1_mean"]),
            "single_std": float(single_row["macro_f1_std"]),
            "pooled_macro_f1": float(pooled_row["macro_f1_mean"]),
            "pooled_std": float(pooled_row["macro_f1_std"]),
            "delta": delta,
            "meets_criterion": delta >= SUCCESS_DELTA,
        }
        if "heldout_macro_f1_mean" in agg and pd.notna(single_row.get("heldout_macro_f1_mean")):
            record["single_heldout"] = float(single_row["heldout_macro_f1_mean"])
            record["single_heldout_std"] = float(single_row["heldout_macro_f1_std"])
            record["delta_heldout"] = float(pooled_row["macro_f1_mean"]) - record["single_heldout"]
        rows.append(record)
    return pd.DataFrame(rows)


def paired_delta(rows: pd.DataFrame) -> pd.DataFrame:
    """Delta pooled − single-source ghép cặp theo seed (ưu tiên held-out)."""
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for source in SOURCES:
        pooled = rows[
            (rows["experiment"] == "rq3_train_pooled_test_each") & (rows["test_source"] == source)
        ]
        single = rows[
            (rows["experiment"] == SINGLE_SOURCE[source]) & (rows["test_source"] == source)
        ]
        if pooled.empty or single.empty:
            continue
        single = single.copy()
        if "heldout_macro_f1" in single.columns:
            single["compare_value"] = single["heldout_macro_f1"].fillna(single["macro_f1"])
        else:
            single["compare_value"] = single["macro_f1"]

        merged = (
            pooled[["seed", "macro_f1"]]
            .rename(columns={"macro_f1": "pooled"})
            .merge(
                single[["seed", "compare_value"]].rename(columns={"compare_value": "single"}),
                on="seed",
                how="inner",
            )
        )
        if merged.empty:
            continue
        deltas = merged["pooled"] - merged["single"]
        std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        records.append(
            {
                "test_source": source,
                "n_pairs": int(len(deltas)),
                "delta_mean": float(deltas.mean()),
                "delta_std": std,
                "delta_ci95": ci95(std, len(deltas)),
                "seeds": ", ".join(
                    f"{int(seed)}: {value:+.3f}"
                    for seed, value in zip(merged["seed"], deltas, strict=True)
                ),
            }
        )
    return pd.DataFrame(records)


def dose_response(agg: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 theo tỉ lệ TrashNet trong train (0, 25, 50, 75, 100%)."""
    if agg.empty:
        return agg
    frame = agg[agg["experiment"].isin(MIX_EXPERIMENTS)].copy()
    frame["trashnet_share"] = frame["experiment"].map(MIX_EXPERIMENTS)
    return (
        frame[
            [
                "trashnet_share",
                "label",
                "test_source",
                "n_seeds",
                "macro_f1_mean",
                "macro_f1_std",
                "n_test",
            ]
        ]
        .sort_values(["test_source", "trashnet_share"])
        .reset_index(drop=True)
    )


def domain_gap(agg: pd.DataFrame) -> pd.DataFrame:
    """Chênh lệch Macro-F1 giữa validation in-domain và test của từng protocol."""
    if agg.empty:
        return agg
    frame = agg[agg["test_source"] != "all"].copy()
    frame["gap"] = frame["best_val_macro_f1"] - frame["macro_f1_mean"]
    return frame[
        ["label", "train", "test_source", "best_val_macro_f1", "macro_f1_mean", "gap"]
    ].reset_index(drop=True)


def per_class_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-class F1 trung bình theo protocol và nguồn test."""
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        for class_name, values in row["per_class"].items():
            records.append(
                {
                    "label": row["label"],
                    "test_source": row["test_source"],
                    "class": class_name,
                    "f1": values["f1"],
                    "recall": values["recall"],
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return (
        frame.groupby(["label", "test_source", "class"], dropna=False)
        .agg(f1_mean=("f1", "mean"), recall_mean=("recall", "mean"))
        .reset_index()
    )


def prediction_shift(data: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    """Tỉ lệ nhãn dự đoán trên test, để thấy hiện tượng sụp về một lớp."""
    records: list[dict[str, Any]] = []
    for experiment, runs in data.items():
        info = PROTOCOLS[experiment]
        for source in info["tests"]:
            counts: Counter[str] = Counter()
            total = 0
            for run in runs:
                predictions = run["predictions"]
                if predictions is None or predictions.empty:
                    continue
                subset = predictions[predictions["dataset"] == source]
                counts.update(subset["pred_label"].tolist())
                total += len(subset)
            if not total:
                continue
            top_class, top_count = counts.most_common(1)[0]
            records.append(
                {
                    "label": info["label"],
                    "test_source": source,
                    "top_predicted": top_class,
                    "top_share": top_count / total,
                    "distribution": {
                        name: counts.get(name, 0) / total for name in CANONICAL_CLASSES
                    },
                }
            )
    return pd.DataFrame(records)


def _confusion_panels(data: dict[str, list[dict[str, Any]]], figures: Path) -> list[str]:
    panels: list[tuple[str, np.ndarray]] = []
    for experiment, runs in data.items():
        info = PROTOCOLS[experiment]
        for source in info["tests"]:
            total = np.zeros((len(CANONICAL_CLASSES), len(CANONICAL_CLASSES)), dtype=float)
            found = False
            for run in runs:
                predictions = run["predictions"]
                if predictions is None or predictions.empty:
                    continue
                subset = predictions[predictions["dataset"] == source]
                if subset.empty:
                    continue
                total += np.asarray(compute_metrics(subset)["confusion_matrix"], dtype=float)
                found = True
            if found:
                panels.append((f"{info['label']} | test {source}", total))
    if not panels:
        return []

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.6 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (title, matrix) in zip(axes_array, panels, strict=False):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = matrix / np.where(row_sums == 0, 1, row_sums)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.set_xticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES, rotation=45, fontsize=7)
        axis.set_yticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES, fontsize=7)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                axis.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=6)
    for axis in axes_array[len(panels) :]:
        axis.axis("off")
    figure.colorbar(image, ax=axes_array.tolist(), fraction=0.02)
    path = figures / "confusion_matrices.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return [path.name]


def _dose_figure(dose: pd.DataFrame, figures: Path) -> list[str]:
    if dose.empty:
        return []
    figure, axis = plt.subplots(figsize=(7, 4.6))
    for source in SOURCES:
        part = dose[dose["test_source"] == source].sort_values("trashnet_share")
        if part.empty:
            continue
        axis.errorbar(
            part["trashnet_share"],
            part["macro_f1_mean"],
            yerr=part["macro_f1_std"],
            marker="o",
            capsize=3,
            label=f"test {source}",
        )
    axis.set_xlabel("Tỉ lệ TrashNet trong train (%)")
    axis.set_ylabel("Macro-F1")
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = figures / "dose_response.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path.name]


def _bar_by_source(agg: pd.DataFrame, figures: Path) -> list[str]:
    subset = agg[agg["test_source"].isin(SOURCES)]
    if subset.empty:
        return []
    figure, axis = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    for index, source in enumerate(SOURCES):
        group = subset[subset["test_source"] == source]
        positions = index + (np.arange(len(group)) - (len(group) - 1) / 2) * width
        axis.bar(
            positions,
            group["macro_f1_mean"],
            width=width,
            yerr=group["macro_f1_std"],
            capsize=4,
            label=source,
        )
        for position, (_, row) in zip(positions, group.iterrows(), strict=True):
            axis.text(position, row["macro_f1_mean"] + 0.02, row["label"], ha="center", fontsize=7)
    axis.set_xticks(range(len(SOURCES)), [f"Test {source}" for source in SOURCES])
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
            elif isinstance(value, (np.floating, np.integer)):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dose_verdict(dose: pd.DataFrame) -> str:
    """Kết luận theo tiêu chí đã đăng ký trong plan source_mix."""
    trashnet = dose[dose["test_source"] == "trashnet"].set_index("trashnet_share")
    realwaste = dose[dose["test_source"] == "realwaste"].set_index("trashnet_share")
    if not {0, 25, 50}.issubset(set(trashnet.index)):
        return "_Chưa đủ mức 0/25/50% để kết luận._"
    try:
        gain = float(trashnet.loc[25, "macro_f1_mean"] - trashnet.loc[0, "macro_f1_mean"])
        at50 = float(trashnet.loc[50, "macro_f1_mean"])
        drop = float(realwaste.loc[0, "macro_f1_mean"] - realwaste.loc[50, "macro_f1_mean"])
    except KeyError:
        return "_Chưa đủ dữ liệu để kết luận._"

    met = gain >= 0.05 and at50 >= 0.80 and drop <= 0.03
    return "\n".join(
        [
            f"- Tăng 0% → 25%: **{gain:+.4f}** Macro-F1 trên TrashNet (tiêu chí ≥ +0.05).",
            f"- Mức 50%: **{at50:.4f}** trên TrashNet (tiêu chí ≥ 0.80).",
            f"- RealWaste giảm khi lên 50%: **{drop:+.4f}** (tiêu chí ≤ 0.03).",
            f"- Tiêu chí plan: **{'đạt' if met else 'chưa đạt'}**.",
        ]
    )


def build_markdown(
    agg: pd.DataFrame,
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    dose: pd.DataFrame,
    per_class: pd.DataFrame,
    shift: pd.DataFrame,
    figures: list[str],
) -> str:
    lines = [
        "# Kết quả RQ3",
        "",
        "Sinh tự động bởi `scripts/analyze_rq3.py` từ artifact của các run `completed`; "
        "test chỉ được đánh giá một lần mỗi run.",
        "",
        f"Tiêu chí: pooled cao hơn single-source ≥ {SUCCESS_DELTA} Macro-F1 trên **cả hai** "
        "nguồn test ngoài miền.",
        "",
    ]
    if agg.empty:
        lines.append("_Chưa có run completed._")
        return "\n".join(lines)

    summary = pd.DataFrame(
        {
            "Protocol": agg["label"],
            "Train": agg["train"],
            "Test": agg["test_source"],
            "n_seeds": agg["n_seeds"],
            "n_test": agg["n_test"],
            "Macro-F1 (mean ± std)": [
                f"{mean:.4f} ± {std:.4f}"
                for mean, std in zip(agg["macro_f1_mean"], agg["macro_f1_std"], strict=True)
            ],
            "CI95 (±)": [
                f"{value:.4f}" if pd.notna(value) else "—" for value in agg["macro_f1_ci95"]
            ],
            "Accuracy": agg["accuracy_mean"].round(4),
            "Balanced acc": agg["balanced_accuracy_mean"].round(4),
            "Val Macro-F1 (in-domain)": agg["best_val_macro_f1"].round(4),
            "Gap val−test": (agg["best_val_macro_f1"] - agg["macro_f1_mean"]).round(4),
        }
    )
    lines.extend(["## Tổng hợp", "", _markdown_table(summary, list(summary.columns)), ""])
    lines.extend(
        [
            "## Lưu ý phương pháp",
            "",
            "- **Single-source:** test là **toàn bộ** nguồn còn lại (TrashNet 2.390 hoặc RealWaste "
            "3.092) — mô hình chưa từng thấy nguồn đó, nên đây là test ngoài miền thuần.",
            "- **Pooled (100%) và các mức mix:** test là phần held-out của split v1 theo từng "
            "nguồn (TrashNet 360, RealWaste 463) — nguồn đó đã xuất hiện trong train ở mức "
            "tương ứng.",
            "- So sánh công bằng dùng cách chấm single-source **trên cùng held-out test** "
            "(mục dưới): chỉ lọc predictions đã có, không đánh giá lại test.",
            "- CI95 tính theo phân phối t với n=3 nên khá rộng. Hiệu số nhỏ hơn CI (ví dụ giữa "
            "các mức mix gần nhau) **không** nên đọc là khác biệt thực.",
            "- Có nhiều biến thể × nhiều chỉ số nhưng không hiệu chỉnh multiple comparisons; "
            "các so sánh ngoài tiêu chí đăng ký chỉ mang tính exploratory.",
            "",
        ]
    )

    if not comparison.empty:
        table = pd.DataFrame(
            {
                "Test source": comparison["test_source"],
                "Single-source": comparison["single_label"],
                "Single (mean ± std)": [
                    f"{mean:.4f} ± {std:.4f}"
                    for mean, std in zip(
                        comparison["single_macro_f1"], comparison["single_std"], strict=True
                    )
                ],
                "Pooled (mean ± std)": [
                    f"{mean:.4f} ± {std:.4f}"
                    for mean, std in zip(
                        comparison["pooled_macro_f1"], comparison["pooled_std"], strict=True
                    )
                ],
                "Delta": comparison["delta"].round(4),
                "Đạt ≥ 0.02": comparison["meets_criterion"].map({True: "có", False: "không"}),
            }
        )
        lines.extend(
            [
                "## Pooled so với single-source",
                "",
                _markdown_table(table, list(table.columns)),
                "",
            ]
        )

        if "single_heldout" in comparison:
            fair = pd.DataFrame(
                {
                    "Test source": comparison["test_source"],
                    "Single (held-out, mean ± std)": [
                        f"{mean:.4f} ± {std:.4f}"
                        for mean, std in zip(
                            comparison["single_heldout"],
                            comparison["single_heldout_std"],
                            strict=True,
                        )
                    ],
                    "Pooled (mean ± std)": table["Pooled (mean ± std)"],
                    "Delta held-out": comparison["delta_heldout"].round(4),
                }
            )
            lines.extend(
                [
                    "### Trên cùng held-out test",
                    "",
                    _markdown_table(fair, list(fair.columns)),
                    "",
                ]
            )

        if not paired.empty:
            paired_table = pd.DataFrame(
                {
                    "Test source": paired["test_source"],
                    "n cặp": paired["n_pairs"],
                    "Delta trung bình": paired["delta_mean"].round(4),
                    "SD": paired["delta_std"].round(4),
                    "CI95 (±)": [
                        f"{value:.4f}" if pd.notna(value) else "—" for value in paired["delta_ci95"]
                    ],
                    "Delta từng seed": paired["seeds"],
                }
            )
            lines.extend(
                [
                    "### Delta ghép cặp theo seed",
                    "",
                    "Cùng seed nên khử được phần lớn phương sai giữa seed; dùng để kiểm tra "
                    "delta có lớn hơn nhiễu hay không (n=3, không báo p-value).",
                    "",
                    _markdown_table(paired_table, list(paired_table.columns)),
                    "",
                ]
            )

        if bool(comparison["meets_criterion"].all()):
            lines.extend(
                [
                    "**Kết luận:** đạt tiêu chí đã đăng ký (delta ≥ 0.02) trên cả hai nguồn. "
                    "Delta trên cùng held-out test xác nhận kết luận không phải do khác kích "
                    "thước test set.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "**Kết luận:** tiêu chí chưa đạt trên mọi nguồn; báo cáo trung thực và "
                    "phân tích domain shift theo kế hoạch dự phòng trong plan.",
                    "",
                ]
            )

    if not dose.empty:
        pivot = dose.pivot_table(
            index="trashnet_share", columns="test_source", values="macro_f1_mean"
        )
        spread = dose.pivot_table(
            index="trashnet_share", columns="test_source", values="macro_f1_std"
        )
        rows = []
        for share in pivot.index:
            row: dict[str, Any] = {"TrashNet %": int(share)}
            for source in SOURCES:
                if source in pivot.columns and pd.notna(pivot.loc[share, source]):
                    row[f"Macro-F1 {source}"] = (
                        f"{pivot.loc[share, source]:.4f} ± {spread.loc[share, source]:.4f}"
                    )
            rows.append(row)
        table = pd.DataFrame(rows)
        lines.extend(
            [
                "## Tỉ lệ dữ liệu nguồn thứ hai (mix-ratio)",
                "",
                "RealWaste giữ 100% train; cột là tỉ lệ group TrashNet train được thêm vào.",
                "",
                _markdown_table(table, list(table.columns)),
                "",
                dose_verdict(dose),
                "",
                "**Lưu ý:** val dùng cả hai nguồn ở mọi mức tỉ lệ. Với mức 0%, val TrashNet là "
                "ngoài miền nên epoch được chọn có thể chưa tối ưu cho RealWaste; đọc kèm cột "
                "Val Macro-F1 và Gap val−test trong bảng tổng hợp.",
                "",
            ]
        )

    if not per_class.empty:
        pivot = per_class.pivot_table(
            index=["label", "test_source"],
            columns="class",
            values="f1_mean",
        ).reset_index()
        pivot.columns = [str(column) for column in pivot.columns]
        lines.extend(
            ["## Per-class F1 trên test", "", _markdown_table(pivot, list(pivot.columns)), ""]
        )

    if not shift.empty:
        table = pd.DataFrame(
            {
                "Protocol": shift["label"],
                "Test": shift["test_source"],
                "Lớp dự đoán nhiều nhất": shift["top_predicted"],
                "Tỉ lệ": shift["top_share"].round(3),
            }
        )
        lines.extend(
            [
                "## Sụp đổ dự đoán (domain shift)",
                "",
                "Tỉ lệ nhãn dự đoán nhiều nhất trên test; tỉ lệ cao cho thấy mô hình dồn về "
                "một lớp khi đổi miền.",
                "",
                _markdown_table(table, list(table.columns)),
                "",
            ]
        )

    if figures:
        lines.extend(["## Hình", ""])
        lines.extend(f"- `{figure}`" for figure in figures)
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phân tích RQ3.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument("--output-md", type=Path, default=Path("docs/RQ3/rq3_results.md"))
    parser.add_argument("--output-csv", type=Path, default=Path("docs/RQ3/rq3_metrics.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/RQ3/rq3_analysis"))
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Split v1 dùng để lấy held-out test khi chấm công bằng (mặc định data/splits/v1).",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    split_csv = args.split_csv or (splits_dir() / "v1" / "split.csv")
    data = load_runs(args.experiments_root.resolve())
    rows = _run_rows(data, heldout_test_paths(split_csv))
    agg = aggregate(rows)
    comparison = compare_pooled(agg)
    paired = paired_delta(rows)
    dose = dose_response(agg)
    per_class = per_class_table(rows)
    shift = prediction_shift(data)

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    figures.extend(_bar_by_source(agg, args.figures_dir))
    figures.extend(_dose_figure(dose, args.figures_dir))
    figures.extend(_confusion_panels(data, args.figures_dir))

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        build_markdown(agg, comparison, paired, dose, per_class, shift, figures) + "\n",
        encoding="utf-8",
    )
    if not agg.empty:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.output_csv, index=False)

    missing = [
        f"{experiment} ({len(runs)} run)" for experiment, runs in data.items() if len(runs) < 3
    ]
    print(f"Đã ghi {args.output_md}")
    print(f"Figures: {args.figures_dir} ({len(figures)} hình)")
    if missing:
        print("Thiếu seed cho: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
