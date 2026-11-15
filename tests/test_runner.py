"""Test tích hợp training engine trên dữ liệu nhỏ."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from src.training.runner import run_training


def make_data(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        for label, color in (("glass", (20, 40, 60)), ("paper", (200, 180, 160))):
            for index in range(2):
                relative = f"raw/fake/{label}/{split}_{index}.jpg"
                image_path = data / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), color=color).save(image_path)
                rows.append(
                    {
                        "path": relative,
                        "dataset": "realwaste",
                        "label": label,
                        "raw_label": label.capitalize(),
                        "split": split,
                        "group_id": f"g_{split}_{label}_{index}",
                    }
                )

    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return data, split_csv


def make_config(
    tmp_path: Path,
    data: Path,
    split_csv: Path,
    *,
    evaluate_test: bool = False,
) -> Path:
    config = {
        "experiment": "rq1_runner_test",
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
            "epochs": 2,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "early_stopping_patience": 5,
            "monitor": "val_macro_f1",
            "amp": False,
            "evaluate_test": evaluate_test,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_training_end_to_end(tmp_path: Path, plan_factory) -> None:
    data, split_csv = make_data(tmp_path)
    config_path = make_config(tmp_path, data, split_csv)
    plan_path = plan_factory(
        baseline="rq1_runner_test",
        variants=[{"experiment": "rq1_runner_other", "change": "x", "fixed": ["y"]}],
        seeds=[42],
    )

    result = run_training(
        config_path=config_path,
        plan_path=plan_path,
        seed=42,
        limit_batches=1,
        experiments_root=tmp_path / "experiments",
    )

    run_dir = result.run_dir
    assert run_dir.is_dir()
    assert (run_dir / "best.pt").is_file() or (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "metrics.json").is_file()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] is None
    assert metrics["best_val"]["macro_f1"] >= 0.0
    assert metrics["class_names"] == ["cardboard", "glass", "metal", "paper", "plastic"]

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed"
    assert run["plan"]["plan_id"] == "rq1_demo"
    assert run["model"]["name"] == "custom_cnn"

    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    epoch_records = [record for record in history if "phase" in record]
    assert len(epoch_records) == 4  # 2 epoch x (train, val)
    val_records = [record for record in epoch_records if record["phase"] == "val"]
    assert "per_class_f1" in val_records[0]


def test_run_training_with_test_evaluation(tmp_path: Path, plan_factory) -> None:
    data, split_csv = make_data(tmp_path)
    config_path = make_config(tmp_path, data, split_csv, evaluate_test=True)
    plan_path = plan_factory(
        baseline="rq1_runner_test",
        variants=[{"experiment": "rq1_runner_other", "change": "x", "fixed": ["y"]}],
        seeds=[42],
    )

    result = run_training(
        config_path=config_path,
        plan_path=plan_path,
        seed=42,
        limit_batches=1,
        experiments_root=tmp_path / "experiments",
    )

    assert result.test is not None
    predictions = result.run_dir / "predictions" / "test.csv"
    assert predictions.is_file()
    rows = list(csv.DictReader(predictions.open(encoding="utf-8")))
    assert len(rows) == 2  # limit 1 batch x batch size 2
    assert "prob_glass" in rows[0]
