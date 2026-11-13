"""Đường dẫn dùng chung, cấu hình được qua biến môi trường.

Local mặc định dùng `data/` và `experiments/` trong repo. Trên Colab có thể trỏ
sang Drive mà không cần sửa code:

    WASTE_DATA_ROOT=/content/data
    WASTE_EXPERIMENTS_ROOT=/content/drive/MyDrive/waste_classifier/experiments
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    path = Path(value).expanduser() if value else default
    return path.resolve()


def project_root() -> Path:
    """Thư mục gốc của repo."""
    return PROJECT_ROOT


def data_root() -> Path:
    """Thư mục dữ liệu (mặc định `data/`)."""
    return _resolve("WASTE_DATA_ROOT", PROJECT_ROOT / "data")


def raw_root() -> Path:
    """Thư mục dữ liệu thô (mặc định `data/raw/`)."""
    return data_root() / "raw"


def dataset_dir(name: str) -> Path:
    """Thư mục của một dataset thô."""
    return raw_root() / name


def splits_dir() -> Path:
    """Thư mục chứa split CSV. Mặc định nằm trong repo để dùng chung."""
    return _resolve("WASTE_SPLITS_DIR", PROJECT_ROOT / "data" / "splits")


def experiments_root() -> Path:
    """Thư mục kết quả thực nghiệm (mặc định `experiments/`)."""
    return _resolve("WASTE_EXPERIMENTS_ROOT", PROJECT_ROOT / "experiments")


def experiment_dir(rq: str, run_id: str) -> Path:
    """Thư mục của một lần chạy."""
    return experiments_root() / rq / run_id


def ensure_dir(path: Path) -> Path:
    """Tạo thư mục nếu chưa có và trả lại chính path đó."""
    path.mkdir(parents=True, exist_ok=True)
    return path
