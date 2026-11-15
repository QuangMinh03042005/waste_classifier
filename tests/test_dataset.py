"""Test dataset và dataloader."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import WasteDataset, create_dataloaders, create_datasets


def make_split_csv(tmp_path: Path, *, extra_rows: list[dict] | None = None) -> Path:
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

    rows.extend(extra_rows or [])
    split_csv = tmp_path / "split.csv"
    with split_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return split_csv


def test_dataset_filters_split(tmp_path: Path) -> None:
    split_csv = make_split_csv(tmp_path)
    dataset = WasteDataset(
        data_root=tmp_path / "data",
        split_csv=split_csv,
        split="train",
    )

    assert len(dataset) == 4
    assert dataset.class_counts == {
        "cardboard": 0,
        "glass": 2,
        "metal": 0,
        "paper": 2,
        "plastic": 0,
    }
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label in (1, 3)
    assert dataset.sample(0).split == "train"


def test_dataset_missing_file(tmp_path: Path) -> None:
    split_csv = make_split_csv(tmp_path)
    (tmp_path / "data" / "raw" / "fake" / "glass" / "train_0.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Thiếu"):
        WasteDataset(data_root=tmp_path / "data", split_csv=split_csv, split="train")


def test_dataset_unknown_label(tmp_path: Path) -> None:
    split_csv = make_split_csv(
        tmp_path,
        extra_rows=[
            {
                "path": "raw/fake/unknown/1.jpg",
                "dataset": "realwaste",
                "label": "unknown",
                "raw_label": "Unknown",
                "split": "train",
                "group_id": "g_unknown",
            }
        ],
    )
    with pytest.raises(ValueError, match="class_names"):
        WasteDataset(data_root=tmp_path / "data", split_csv=split_csv, split="train")


def test_create_datasets_and_loaders(tmp_path: Path) -> None:
    split_csv = make_split_csv(tmp_path)
    datasets = create_datasets(
        data_root=tmp_path / "data",
        split_csv=split_csv,
        image_size=64,
    )
    assert set(datasets) == {"train", "val", "test"}
    assert len(datasets["val"]) == 4

    loaders = create_dataloaders(
        data_root=tmp_path / "data",
        split_csv=split_csv,
        image_size=64,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=42,
    )
    images, labels = next(iter(loaders["train"]))
    assert images.shape == (2, 3, 64, 64)
    assert labels.shape == (2,)
    assert images.dtype == torch.float32
