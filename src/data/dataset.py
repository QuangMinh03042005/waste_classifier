"""Dataset và DataLoader cho bài toán phân loại rác thải.

Split được đọc từ `data/splits/v1/split.csv`; đường dẫn trong CSV tương đối so
với data root nên dùng được ở local và Colab.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.labels import CANONICAL_CLASSES
from src.data.split import SplitRow, read_split_csv
from src.data.transforms import DEFAULT_IMAGE_SIZE, build_transforms

SPLITS = ("train", "val", "test")


class WasteDataset(Dataset):
    """Dataset cho một split, trả về (image_tensor, label_index)."""

    def __init__(
        self,
        *,
        data_root: Path,
        split_csv: Path,
        split: str,
        class_names: Sequence[str] = CANONICAL_CLASSES,
        transform=None,
        validate_paths: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split phải thuộc {SPLITS}, nhận {split!r}")

        self.data_root = Path(data_root)
        self.split_csv = Path(split_csv)
        self.split = split
        self.class_names = tuple(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}

        rows = read_split_csv(self.split_csv)
        self.samples = [row for row in rows if row.split == split]
        if not self.samples:
            raise ValueError(f"Split {split!r} không có ảnh nào trong {self.split_csv}")

        unknown = sorted({row.label for row in self.samples} - set(self.class_names))
        if unknown:
            raise ValueError(f"Nhãn không có trong class_names: {unknown}")

        self.transform = (
            transform
            if transform is not None
            else build_transforms(train=split == "train")
        )

        if validate_paths:
            self._validate_paths()

    def _validate_paths(self) -> None:
        missing = [
            str(self.data_root / row.path)
            for row in self.samples
            if not (self.data_root / row.path).is_file()
        ]
        if missing:
            preview = "\n  ".join(missing[:5])
            raise FileNotFoundError(
                f"Thiếu {len(missing)} ảnh trong {self.data_root}. Ví dụ:\n  {preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image_path = self.data_root / sample.path
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Không đọc được ảnh {image_path}: {exc}") from exc

        tensor = self.transform(rgb)
        return tensor, self.class_to_index[sample.label]

    def sample(self, index: int) -> SplitRow:
        """Metadata của mẫu, dùng khi ghi predictions."""
        return self.samples[index]

    @property
    def class_counts(self) -> dict[str, int]:
        counts = Counter(row.label for row in self.samples)
        return {name: counts.get(name, 0) for name in self.class_names}

    @property
    def dataset_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(row.dataset for row in self.samples).items()))


def create_datasets(
    *,
    data_root: Path,
    split_csv: Path,
    class_names: Sequence[str] = CANONICAL_CLASSES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augmentation: bool = True,
    validate_paths: bool = True,
) -> dict[str, WasteDataset]:
    """Tạo dataset cho cả 3 split với transform tương ứng."""
    return {
        split: WasteDataset(
            data_root=data_root,
            split_csv=split_csv,
            split=split,
            class_names=class_names,
            transform=build_transforms(
                train=split == "train",
                image_size=image_size,
                augmentation=augmentation,
            ),
            validate_paths=validate_paths,
        )
        for split in SPLITS
    }


def create_dataloaders(
    *,
    data_root: Path,
    split_csv: Path,
    class_names: Sequence[str] = CANONICAL_CLASSES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 64,
    num_workers: int = 0,
    augmentation: bool = True,
    pin_memory: bool | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Tạo DataLoader cho train/val/test."""
    if batch_size <= 0:
        raise ValueError(f"batch_size phải > 0, nhận {batch_size}")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    datasets = create_datasets(
        data_root=data_root,
        split_csv=split_csv,
        class_names=class_names,
        image_size=image_size,
        augmentation=augmentation,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        is_train = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=generator if is_train else None,
            persistent_workers=num_workers > 0,
        )
    return loaders
