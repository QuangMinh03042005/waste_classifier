"""Test evaluator độc lập cho RQ3."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.models import build_model
from src.training.checkpoint import save_best
from src.training.evaluator import evaluate_checkpoint


def make_data(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    rows = []
    for label, color in (("glass", (20, 40, 60)), ("paper", (200, 180, 160))):
        for index in range(2):
            relative = f"raw/fake/{label}/{index}.jpg"
            image_path = data / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 48), color=color).save(image_path)
            rows.append(
                {
                    "path": relative,
                    "dataset": "realwaste",
                    "label": label,
                    "raw_label": label.capitalize(),
                    "split": "test",
                    "group_id": f"g_{label}_{index}",
                }
            )
    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data, split_csv


def make_config(data: Path, split_csv: Path) -> dict:
    return {
        "experiment": "eval_test",
        "rq": "rq1",
        "seed": 42,
        "data": {
            "root": str(data),
            "split_csv": str(split_csv),
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "class_names": ["cardboard", "glass", "metal", "paper", "plastic"],
        },
        "model": {
            "name": "custom_cnn",
            "num_classes": 5,
            "channels": [8, 16, 32, 64],
            "convs_per_stage": [1, 1, 1, 1],
            "dropout": 0.1,
        },
        "training": {"epochs": 1},
    }


def test_evaluate_checkpoint(tmp_path: Path) -> None:
    data, split_csv = make_data(tmp_path)
    config = make_config(data, split_csv)
    model = build_model("custom_cnn", **{k: v for k, v in config["model"].items() if k != "name"})
    checkpoint = tmp_path / "best.pt"
    save_best(
        checkpoint,
        model,
        epoch=1,
        metric=0.5,
        class_names=config["data"]["class_names"],
        config=config,
        seed=42,
    )

    result = evaluate_checkpoint(
        config=config,
        checkpoint=checkpoint,
        split="test",
        datasets=None,
        output_dir=tmp_path / "eval",
    )

    assert result.metrics["n_samples"] == 4
    assert 0.0 <= result.metrics["macro_f1"] <= 1.0
    assert result.predictions_path.is_file()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["checkpoint"] == str(checkpoint)
    rows = list(csv.DictReader(result.predictions_path.open(encoding="utf-8")))
    assert len(rows) == 4
