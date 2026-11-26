"""So sánh Grad-CAM của nhiều model trên cùng một bộ ảnh."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.data.labels import CANONICAL_CLASSES
from src.data.transforms import build_transforms
from src.training.checkpoint import load_best
from src.training.efficiency import build_model_from_config
from src.training.gradcam import GradCAM, overlay_cam
from src.utils.paths import data_root as default_data_root
from src.utils.paths import project_root


@dataclass
class RunBundle:
    """Một model đã load cùng predictions tương ứng."""

    label: str
    run_dir: Path
    model: torch.nn.Module
    predictions: pd.DataFrame
    class_names: list[str]
    data_root: Path
    image_size: int
    transform: Any
    display: Any
    device: torch.device


@dataclass
class ImageCase:
    """Một ảnh được chọn để so sánh."""

    path: str
    true_label: str
    predictions: dict[str, str]
    confidences: dict[str, float]
    correct: dict[str, bool]
    source: str


def _resolve_data_root(config: dict[str, Any]) -> Path:
    configured = (config.get("data") or {}).get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else project_root() / path
    return default_data_root()


def _prediction_path(run_dir: Path, source: str) -> Path:
    if source == "gd":
        return run_dir / "evaluations" / "external_gd" / "predictions.csv"
    return run_dir / "predictions" / "test.csv"


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    probability_columns = [column for column in frame.columns if column.startswith("prob_")]
    frame["confidence"] = frame[probability_columns].max(axis=1)
    return frame


def load_bundle(run_dir: Path, label: str, *, device: torch.device) -> RunBundle:
    """Load model, transform và predictions của một run."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run.get("config") or {}
    class_names = list(
        (config.get("data") or {}).get("class_names") or CANONICAL_CLASSES
    )
    image_size = int((config.get("data") or {}).get("image_size", 224))
    model = build_model_from_config(config)
    load_best(run_dir / "checkpoints" / "best.pt", model, map_location=device)
    model.to(device).eval()
    return RunBundle(
        label=label,
        run_dir=run_dir,
        model=model,
        predictions=pd.DataFrame(),
        class_names=class_names,
        data_root=_resolve_data_root(config),
        image_size=image_size,
        transform=build_transforms(train=False, image_size=image_size),
        display=transforms.Compose(
            [transforms.Resize(round(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
        ),
        device=device,
    )


def _prepared(bundle: RunBundle, source: str) -> pd.DataFrame:
    frame = _read_predictions(_prediction_path(bundle.run_dir, source))
    return frame[["path", "true_label", "pred_label", "correct", "confidence"]]


def merge_predictions(
    bundles: list[RunBundle], source: str
) -> pd.DataFrame:
    """Gộp dự đoán của nhiều model theo path."""
    merged: pd.DataFrame | None = None
    for bundle in bundles:
        frame = _prepared(bundle, source).rename(
            columns={
                "pred_label": f"pred::{bundle.label}",
                "correct": f"correct::{bundle.label}",
                "confidence": f"confidence::{bundle.label}",
            }
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame,
                on=["path", "true_label"],
                how="inner",
            )
    if merged is None or merged.empty:
        raise ValueError(f"Không ghép được predictions cho source={source}")
    return merged


def _make_case(row: pd.Series, bundles: list[RunBundle], source: str) -> ImageCase:
    predictions = {bundle.label: str(row[f"pred::{bundle.label}"]) for bundle in bundles}
    confidences = {
        bundle.label: float(row[f"confidence::{bundle.label}"]) for bundle in bundles
    }
    correct = {bundle.label: bool(row[f"correct::{bundle.label}"]) for bundle in bundles}
    return ImageCase(
        path=str(row["path"]),
        true_label=str(row["true_label"]),
        predictions=predictions,
        confidences=confidences,
        correct=correct,
        source=source,
    )


def select_cases(
    merged: pd.DataFrame,
    bundles: list[RunBundle],
    class_names: list[str],
    *,
    source: str,
    per_class: int = 2,
) -> tuple[list[ImageCase], list[ImageCase]]:
    """Chọn ảnh đúng/lỗi đại diện mỗi class và ảnh lỗi cho hình summary."""
    selected: list[ImageCase] = []
    for class_name in class_names:
        candidates = merged[merged["true_label"] == class_name]
        if candidates.empty:
            continue
        correctness = np.column_stack(
            [candidates[f"correct::{bundle.label}"].to_numpy(dtype=bool) for bundle in bundles]
        )
        work = candidates.copy()
        work["_all_correct"] = correctness.all(axis=1)
        work["_wrong_count"] = (~correctness).sum(axis=1)
        work["_mean_confidence"] = np.mean(
            np.column_stack(
                [
                    candidates[f"confidence::{bundle.label}"].to_numpy(dtype=float)
                    for bundle in bundles
                ]
            ),
            axis=1,
        )
        correct_rows = work[work["_all_correct"]].sort_values(
            "_mean_confidence", ascending=False
        )
        wrong_rows = work[~work["_all_correct"]].sort_values(
            ["_wrong_count", "_mean_confidence"], ascending=False
        )
        for frame in (correct_rows.head(1), wrong_rows.head(1)):
            if frame.empty:
                continue
            selected.append(_make_case(frame.iloc[0], bundles, source))

    if len(selected) < per_class * len(class_names):
        # fallback: bổ sung ảnh đúng còn lại
        for class_name in class_names:
            candidates = merged[merged["true_label"] == class_name]
            if candidates.empty:
                continue
            correctness = np.column_stack(
                [
                    candidates[f"correct::{bundle.label}"].to_numpy(dtype=bool)
                    for bundle in bundles
                ]
            )
            for index in np.where(correctness.all(axis=1))[0]:
                case = _make_case(candidates.iloc[index], bundles, source)
                if all(existing.path != case.path for existing in selected):
                    selected.append(case)
    selected = selected[: per_class * len(class_names)]

    wrong_cases = []
    for _, frame_row in merged.iterrows():
        correctness = np.array(
            [bool(frame_row[f"correct::{bundle.label}"]) for bundle in bundles]
        )
        if correctness.all():
            continue
        wrong_cases.append(_make_case(frame_row, bundles, source))
    wrong_cases.sort(
        key=lambda case: (
            -sum(not value for value in case.correct.values()),
            -float(np.mean(list(case.confidences.values()))),
        )
    )
    return selected, wrong_cases[:3]


def _plot_grid(
    cases: list[ImageCase],
    bundles: list[RunBundle],
    *,
    title: str,
    output: Path,
) -> Path:
    rows = len(cases)
    columns = len(bundles) + 1
    if rows == 0:
        raise ValueError("Không có ảnh để vẽ")
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows), squeeze=False)
    cams = [GradCAM(bundle.model) for bundle in bundles]
    try:
        for row_index, case in enumerate(cases):
            path = bundles[0].data_root / case.path
            image = Image.open(path).convert("RGB")
            display = bundles[0].display(image)
            axes[row_index][0].imshow(display)
            axes[row_index][0].set_title(f"Thật: {case.true_label}", fontsize=9)
            axes[row_index][0].axis("off")
            for column_index, (bundle, cam) in enumerate(zip(bundles, cams, strict=True), start=1):
                tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)
                predicted = case.predictions[bundle.label]
                target_index = bundle.class_names.index(predicted)
                heatmap, _ = cam.generate(tensor, target_index)
                overlay = overlay_cam(bundle.display(image), heatmap)
                axis = axes[row_index][column_index]
                axis.imshow(overlay)
                axis.set_title(
                    f"{bundle.label}\n{predicted} ({case.confidences[bundle.label]:.2f})",
                    fontsize=8,
                )
                axis.axis("off")
    finally:
        for cam in cams:
            cam.remove()
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def run_comparison(
    runs: dict[str, Path],
    *,
    sources: tuple[str, ...] = ("test", "gd"),
    output_dir: Path,
    per_class: int = 2,
    device: torch.device | None = None,
) -> dict[str, dict[str, Any]]:
    """Sinh hình so sánh Grad-CAM cho từng nguồn."""
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles = [load_bundle(path, label, device=target) for label, path in runs.items()]
    class_names = bundles[0].class_names
    results: dict[str, dict[str, Any]] = {}
    for source in sources:
        merged = merge_predictions(bundles, source)
        selected, wrong_cases = select_cases(
            merged, bundles, class_names, source=source, per_class=per_class
        )
        source_dir = output_dir / source
        summary_path = _plot_grid(
            wrong_cases or selected[:3],
            bundles,
            title=f"Grad-CAM — {source} — ảnh lỗi tiêu biểu",
            output=source_dir / "summary.png",
        )
        class_paths = {}
        for class_name in class_names:
            cases = [case for case in selected if case.true_label == class_name]
            if not cases:
                continue
            class_paths[class_name] = str(
                _plot_grid(
                    cases,
                    bundles,
                    title=f"Grad-CAM — {source} — {class_name}",
                    output=source_dir / f"{class_name}.png",
                )
            )
        results[source] = {
            "summary": str(summary_path),
            "classes": class_paths,
            "selected": pd.DataFrame(
                [
                    {
                        "path": case.path,
                        "true_label": case.true_label,
                        "source": source,
                        **{f"pred::{key}": value for key, value in case.predictions.items()},
                        **{f"conf::{key}": value for key, value in case.confidences.items()},
                    }
                    for case in selected
                ]
            ),
        }
    return results
