"""Test mapping nhãn."""

from __future__ import annotations

import pytest

from src.data.labels import (
    CANONICAL_CLASSES,
    EXCLUDED_CLASSES,
    LABEL_MAPPING,
    LabelMappingError,
    canonical_label,
    class_index,
    display_name,
)


def test_canonical_classes() -> None:
    assert CANONICAL_CLASSES == ("cardboard", "glass", "metal", "paper", "plastic")


def test_realwaste_mapping() -> None:
    assert canonical_label("realwaste", "Cardboard") == "cardboard"
    assert canonical_label("realwaste", "Plastic") == "plastic"
    assert canonical_label("realwaste", "Food Organics") is None
    assert canonical_label("realwaste", "Miscellaneous Trash") is None


def test_trashnet_mapping() -> None:
    assert canonical_label("trashnet", "glass") == "glass"
    assert canonical_label("trashnet", "trash") is None


def test_every_raw_label_is_mapped_or_excluded() -> None:
    for dataset, mapping in LABEL_MAPPING.items():
        for raw_label, label in mapping.items():
            if label is None:
                assert raw_label in EXCLUDED_CLASSES[dataset]
            else:
                assert label in CANONICAL_CLASSES


def test_unknown_label_raises() -> None:
    with pytest.raises(LabelMappingError, match="không có trong mapping"):
        canonical_label("realwaste", "Unknown")
    with pytest.raises(LabelMappingError, match="chưa có mapping"):
        canonical_label("other", "Glass")


def test_class_index_and_display() -> None:
    assert class_index()["glass"] == 1
    assert display_name("glass") == "Glass"
    assert display_name("unknown") == "unknown"
