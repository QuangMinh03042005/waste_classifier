"""Mapping nhãn gốc của từng dataset về 5 lớp chung.

Quy ước: tên lớp chuẩn dùng chữ thường để ổn định giữa Windows/Linux và khi
lưu vào CSV/JSON. Tên hiển thị trong báo cáo có thể viết hoa.
"""

from __future__ import annotations

CANONICAL_CLASSES: tuple[str, ...] = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
)

# Giá trị None = loại khỏi phạm vi 5 lớp.
LABEL_MAPPING: dict[str, dict[str, str | None]] = {
    "realwaste": {
        "Cardboard": "cardboard",
        "Glass": "glass",
        "Metal": "metal",
        "Paper": "paper",
        "Plastic": "plastic",
        "Food Organics": None,
        "Miscellaneous Trash": None,
        "Textile Trash": None,
        "Vegetation": None,
    },
    "trashnet": {
        "cardboard": "cardboard",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": None,
    },
}

EXCLUDED_CLASSES: dict[str, dict[str, str]] = {
    "realwaste": {
        "Food Organics": "không có lớp tương ứng ở TrashNet",
        "Miscellaneous Trash": "không đồng nhất về ngữ nghĩa với 'trash' của TrashNet",
        "Textile Trash": "không có lớp tương ứng ở TrashNet",
        "Vegetation": "không có lớp tương ứng ở TrashNet",
    },
    "trashnet": {
        "trash": "lớp gom nhóm không đồng nhất, không map với RealWaste",
    },
}

DISPLAY_NAMES: dict[str, str] = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
}


class LabelMappingError(ValueError):
    """Nhãn gốc không có trong bảng mapping."""


def canonical_label(dataset: str, raw_label: str) -> str | None:
    """Trả về nhãn chuẩn, hoặc None nếu lớp bị loại."""
    if dataset not in LABEL_MAPPING:
        raise LabelMappingError(
            f"Dataset {dataset!r} chưa có mapping. Hỗ trợ: {sorted(LABEL_MAPPING)}"
        )
    mapping = LABEL_MAPPING[dataset]
    if raw_label not in mapping:
        raise LabelMappingError(
            f"Nhãn {raw_label!r} không có trong mapping của {dataset}. "
            "Kiểm tra lại tên thư mục hoặc cập nhật src/data/labels.py."
        )
    return mapping[raw_label]


def class_index() -> dict[str, int]:
    """Ánh xạ nhãn chuẩn sang chỉ số cố định."""
    return {name: index for index, name in enumerate(CANONICAL_CLASSES)}


def display_name(label: str) -> str:
    """Tên hiển thị cho báo cáo."""
    return DISPLAY_NAMES.get(label, label)
