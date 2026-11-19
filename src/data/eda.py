"""EDA cho bộ dữ liệu 5 lớp: thống kê, ảnh mẫu, chất lượng ảnh và domain shift."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from src.data.labels import CANONICAL_CLASSES, LABEL_MAPPING
from src.data.manifest import DATASETS
from src.data.split import read_split_csv

BRIGHTNESS_LOW = 45.0
BRIGHTNESS_HIGH = 220.0
BLUR_LOW = 25.0
MIN_SIDE = 128


def load_records(split_csv: Path) -> pd.DataFrame:
    rows = read_split_csv(split_csv)
    return pd.DataFrame([asdict(row) for row in rows])


def raw_label_counts() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, spec in DATASETS.items():
        for raw_label, count in spec.expected_counts.items():
            canonical = LABEL_MAPPING[dataset].get(raw_label)
            rows.append(
                {
                    "dataset": dataset,
                    "raw_label": raw_label,
                    "count": count,
                    "canonical": canonical,
                    "used": canonical is not None,
                }
            )
    return pd.DataFrame(rows)


def _image_stats(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        fmt = image.format or ""
        gray = image.convert("L").resize((128, 128))
        pixels = np.asarray(gray, dtype=np.float32)
    laplacian = ndimage.laplace(pixels)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else np.nan,
        "min_side": min(width, height),
        "file_size_kb": path.stat().st_size / 1024,
        "mode": mode,
        "format": fmt,
        "brightness": float(pixels.mean()),
        "contrast": float(pixels.std()),
        "blur": float(laplacian.var()),
    }


def collect_image_stats(
    records: pd.DataFrame,
    data_root: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    selected = records if limit is None else records.head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="EDA images"):
        path = data_root / row["path"]
        try:
            stats = _image_stats(path)
        except (OSError, ValueError) as exc:
            stats = {
                "width": np.nan,
                "height": np.nan,
                "aspect_ratio": np.nan,
                "min_side": np.nan,
                "file_size_kb": np.nan,
                "mode": "",
                "format": "",
                "brightness": np.nan,
                "contrast": np.nan,
                "blur": np.nan,
                "error": str(exc),
            }
        rows.append({**row.to_dict(), **stats})
    return pd.DataFrame(rows)


def class_table(stats: pd.DataFrame) -> pd.DataFrame:
    table = (
        stats.groupby(["label", "dataset"]).size().unstack(fill_value=0).reset_index()
    )
    for dataset in ("realwaste", "trashnet"):
        if dataset not in table:
            table[dataset] = 0
    table["total"] = table[["realwaste", "trashnet"]].sum(axis=1)
    return table.sort_values("total", ascending=False)


def split_table(stats: pd.DataFrame) -> pd.DataFrame:
    return (
        stats.groupby(["split", "label"]).size().unstack(fill_value=0).reset_index()
    )


def domain_summary(stats: pd.DataFrame) -> pd.DataFrame:
    summary = (
        stats.groupby("dataset")
        .agg(
            images=("path", "count"),
            width_mean=("width", "mean"),
            height_mean=("height", "mean"),
            aspect_mean=("aspect_ratio", "mean"),
            aspect_std=("aspect_ratio", "std"),
            brightness_mean=("brightness", "mean"),
            brightness_std=("brightness", "std"),
            blur_mean=("blur", "mean"),
            blur_median=("blur", "median"),
            file_size_kb_mean=("file_size_kb", "mean"),
        )
        .reset_index()
    )
    return summary


def duplicate_table(stats: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        stats.groupby("group_id")
        .agg(
            images=("path", "count"),
            labels=("label", lambda values: ",".join(sorted(set(values)))),
            datasets=("dataset", lambda values: ",".join(sorted(set(values)))),
            splits=("split", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return grouped[grouped["images"] > 1].sort_values("images", ascending=False)


def quality_issues(stats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    duplicates = duplicate_table(stats)
    label_conflicts = duplicates[duplicates["labels"].str.contains(",")]
    return {
        "label_conflicts": label_conflicts,
        "too_dark": stats[stats["brightness"] < BRIGHTNESS_LOW].sort_values("brightness"),
        "too_bright": stats[stats["brightness"] > BRIGHTNESS_HIGH].sort_values(
            "brightness", ascending=False
        ),
        "blurry": stats[stats["blur"] < BLUR_LOW].sort_values("blur"),
        "small": stats[stats["min_side"] < MIN_SIDE].sort_values("min_side"),
        "unreadable": stats[stats["width"].isna()],
    }


def sample_rows(
    stats: pd.DataFrame,
    *,
    per_class: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for dataset in sorted(stats["dataset"].unique()):
        for label in CANONICAL_CLASSES:
            subset = stats[(stats["dataset"] == dataset) & (stats["label"] == label)]
            if subset.empty:
                continue
            count = min(per_class, len(subset))
            indices = rng.choice(subset.index.to_numpy(), size=count, replace=False)
            selected.append(subset.loc[indices])
    return pd.concat(selected, ignore_index=True)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_class_distribution(stats: pd.DataFrame, path: Path) -> Path:
    table = class_table(stats)
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(table))
    width = 0.38
    axis.bar(positions - width / 2, table.get("realwaste", 0), width, label="RealWaste")
    axis.bar(positions + width / 2, table.get("trashnet", 0), width, label="TrashNet")
    axis.set_xticks(positions, table["label"], rotation=15)
    axis.set_ylabel("Số ảnh")
    axis.set_title("Phân bố lớp theo nguồn dữ liệu")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def plot_split_distribution(stats: pd.DataFrame, path: Path) -> Path:
    table = split_table(stats).set_index("split")
    figure, axis = plt.subplots(figsize=(9, 5))
    table.T.plot(kind="bar", ax=axis, width=0.8)
    axis.set_ylabel("Số ảnh")
    axis.set_title("Phân bố lớp theo split")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="split")
    return _save(figure, path)


def plot_sample_grid(
    stats: pd.DataFrame,
    data_root: Path,
    path: Path,
    *,
    dataset: str,
    per_class: int = 6,
) -> Path:
    subset = stats[stats["dataset"] == dataset]
    labels = [label for label in CANONICAL_CLASSES if label in set(subset["label"])]
    figure, axes = plt.subplots(
        len(labels), per_class, figsize=(2.0 * per_class, 2.0 * len(labels))
    )
    if len(labels) == 1:
        axes = np.array([axes])
    for row_index, label in enumerate(labels):
        samples = subset[subset["label"] == label].head(per_class)
        for col_index in range(per_class):
            axis = axes[row_index, col_index]
            axis.axis("off")
            if col_index >= len(samples):
                continue
            row = samples.iloc[col_index]
            with Image.open(data_root / row["path"]) as image:
                axis.imshow(image.convert("RGB"))
            if col_index == 0:
                axis.set_ylabel(label)
                axis.axis("on")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_ylabel(label, fontsize=10)
    figure.suptitle(f"Ảnh mẫu – {dataset}")
    return _save(figure, path)


def plot_image_properties(stats: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].hist(stats["width"].dropna(), bins=30)
    axes[0, 0].set_title("Chiều rộng (px)")
    axes[0, 1].hist(stats["height"].dropna(), bins=30)
    axes[0, 1].set_title("Chiều cao (px)")
    axes[1, 0].hist(stats["brightness"].dropna(), bins=30)
    axes[1, 0].set_title("Độ sáng trung bình")
    axes[1, 1].hist(stats["blur"].dropna(), bins=50)
    axes[1, 1].set_title("Độ nét (Laplacian variance)")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    return _save(figure, path)


def plot_domain_shift(stats: pd.DataFrame, path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, column, title in zip(
        axes.ravel(),
        ["brightness", "blur", "aspect_ratio", "file_size_kb"],
        ["Độ sáng", "Độ nét", "Aspect ratio", "Dung lượng (KB)"],
        strict=True,
    ):
        stats.boxplot(column=column, by="dataset", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.grid(alpha=0.2)
    figure.suptitle("Khác biệt domain giữa 2 nguồn")
    return _save(figure, path)


def plot_duplicate_examples(
    stats: pd.DataFrame,
    data_root: Path,
    duplicates: pd.DataFrame,
    path: Path,
    *,
    max_groups: int = 4,
) -> Path | None:
    groups = duplicates.head(max_groups)
    if groups.empty:
        return None
    figure, axes = plt.subplots(len(groups), 3, figsize=(9, 2.8 * len(groups)))
    if len(groups) == 1:
        axes = np.array([axes])
    for row_index, (_, group) in enumerate(groups.iterrows()):
        members = stats[stats["group_id"] == group["group_id"]].head(3)
        for col_index in range(3):
            axis = axes[row_index, col_index]
            axis.axis("off")
            if col_index >= len(members):
                continue
            row = members.iloc[col_index]
            with Image.open(data_root / row["path"]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{row['label']} | {row['split']}", fontsize=8)
    figure.suptitle("Ví dụ ảnh trùng/near-duplicate theo pHash")
    return _save(figure, path)


def plot_quality_scatter(stats: pd.DataFrame, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    for dataset in sorted(stats["dataset"].unique()):
        subset = stats[stats["dataset"] == dataset]
        axis.scatter(subset["brightness"], subset["blur"], s=8, alpha=0.4, label=dataset)
    axis.set_xlabel("Độ sáng")
    axis.set_ylabel("Độ nét")
    axis.set_title("Phân bố chất lượng ảnh")
    axis.legend()
    axis.grid(alpha=0.2)
    return _save(figure, path)


def build_summary(stats: pd.DataFrame, duplicates: pd.DataFrame) -> dict[str, Any]:
    counts = stats["label"].value_counts()
    issues = quality_issues(stats)
    return {
        "total": int(len(stats)),
        "by_split": stats["split"].value_counts().sort_index().to_dict(),
        "by_dataset": stats["dataset"].value_counts().sort_index().to_dict(),
        "by_class": counts.sort_index().to_dict(),
        "imbalance_ratio": float(counts.max() / counts.min()) if len(counts) > 1 else None,
        "duplicate_groups": int(len(duplicates)),
        "duplicate_images": int(duplicates["images"].sum() - len(duplicates))
        if not duplicates.empty
        else 0,
        "quality_issues": {name: int(len(frame)) for name, frame in issues.items()},
        "thresholds": {
            "brightness_low": BRIGHTNESS_LOW,
            "brightness_high": BRIGHTNESS_HIGH,
            "blur_low": BLUR_LOW,
            "min_side": MIN_SIDE,
        },
    }


def save_tables(
    stats: pd.DataFrame,
    output_dir: Path,
    *,
    data_root: Path,
) -> dict[str, Path]:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    stats.to_csv(tables / "image_stats.csv", index=False)
    paths["image_stats"] = tables / "image_stats.csv"

    raw = raw_label_counts()
    raw.to_csv(tables / "raw_label_counts.csv", index=False)
    paths["raw_label_counts"] = tables / "raw_label_counts.csv"

    class_counts = class_table(stats)
    class_counts.to_csv(tables / "class_counts.csv", index=False)
    paths["class_counts"] = tables / "class_counts.csv"

    splits = split_table(stats)
    splits.to_csv(tables / "split_counts.csv", index=False)
    paths["split_counts"] = tables / "split_counts.csv"

    domain = domain_summary(stats)
    domain.to_csv(tables / "domain_summary.csv", index=False)
    paths["domain_summary"] = tables / "domain_summary.csv"

    duplicates = duplicate_table(stats)
    duplicates.to_csv(tables / "duplicates.csv", index=False)
    paths["duplicates"] = tables / "duplicates.csv"

    issues = quality_issues(stats)
    issue_rows = []
    for name, frame in issues.items():
        for _, row in frame.head(20).iterrows():
            issue_rows.append({"issue": name, **row.to_dict()})
    issues_frame = pd.DataFrame(issue_rows)
    issues_frame.to_csv(tables / "quality_issues.csv", index=False)
    paths["quality_issues"] = tables / "quality_issues.csv"

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(build_summary(stats, duplicates), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["summary"] = summary_path
    return paths


def generate_all(
    stats: pd.DataFrame,
    data_root: Path,
    output_dir: Path,
) -> dict[str, Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["class_distribution"] = plot_class_distribution(
        stats, figures / "class_distribution.png"
    )
    paths["split_distribution"] = plot_split_distribution(
        stats, figures / "split_distribution.png"
    )
    paths["sample_realwaste"] = plot_sample_grid(
        stats, data_root, figures / "sample_realwaste.png", dataset="realwaste"
    )
    paths["sample_trashnet"] = plot_sample_grid(
        stats, data_root, figures / "sample_trashnet.png", dataset="trashnet"
    )
    paths["image_properties"] = plot_image_properties(
        stats, figures / "image_properties.png"
    )
    paths["domain_shift"] = plot_domain_shift(stats, figures / "domain_shift.png")
    paths["quality_scatter"] = plot_quality_scatter(
        stats, figures / "quality_scatter.png"
    )
    duplicates = duplicate_table(stats)
    duplicate_plot = plot_duplicate_examples(
        stats, data_root, duplicates, figures / "duplicates.png"
    )
    if duplicate_plot is not None:
        paths["duplicates"] = duplicate_plot
    return paths
