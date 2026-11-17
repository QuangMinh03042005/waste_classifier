"""Test batch runner skip/resume."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml
from PIL import Image

from scripts.run_batch import main


def make_data(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    rows = []
    for split in ("train", "val", "test"):
        for label, color in (("glass", (20, 40, 60)), ("paper", (200, 180, 160))):
            relative = f"raw/fake/{label}/{split}.jpg"
            image_path = data / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=color).save(image_path)
            rows.append(
                {
                    "path": relative,
                    "dataset": "realwaste",
                    "label": label,
                    "raw_label": label.capitalize(),
                    "split": split,
                    "group_id": f"g_{split}_{label}",
                }
            )
    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data, split_csv


def make_config(tmp_path: Path, data: Path, split_csv: Path) -> Path:
    config = {
        "experiment": "rq1_batch_test",
        "rq": "rq1",
        "seed": 42,
        "data": {
            "root": str(data),
            "split_csv": str(split_csv),
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "augmentation": False,
            "class_weighting": False,
        },
        "model": {
            "name": "custom_cnn",
            "num_classes": 5,
            "channels": [8, 16, 32, 64],
            "convs_per_stage": [1, 1, 1, 1],
            "dropout": 0.1,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "early_stopping_patience": 2,
            "amp": False,
            "evaluate_test": False,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_batch_skips_completed(tmp_path: Path, plan_factory) -> None:
    data, split_csv = make_data(tmp_path)
    config = make_config(tmp_path, data, split_csv)
    plan = plan_factory(
        baseline="rq1_batch_test",
        variants=[{"experiment": "rq1_batch_other", "change": "x", "fixed": ["y"]}],
        seeds=[42],
    )
    experiments = tmp_path / "experiments"

    args = [
        "--configs",
        str(config),
        "--plan",
        str(plan),
        "--seeds",
        "42",
        "--experiments-root",
        str(experiments),
        "--epochs",
        "1",
        "--limit-batches",
        "1",
    ]
    assert main(args) == 0
    run_dir = experiments / "rq1" / "rq1_batch_test_s42"
    assert (run_dir / "metrics.json").is_file()

    assert main(args) == 0
    assert (run_dir / "metrics.json").is_file()
