"""Đóng gói dữ liệu thô thành một file zip để đưa lên Google Drive.

Bundle giữ tiền tố `raw/`, nên khi giải nén vào `/content/data` sẽ cho
`/content/data/raw/<dataset>/...` đúng layout mà pipeline mong đợi.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.data.download import safe_extract, sha256_file
from src.data.manifest import DATASETS

EXCLUDED_DIRS = {"_downloads", ".tmp"}


@dataclass(frozen=True, slots=True)
class BundleInfo:
    path: Path
    sha256: str
    bytes: int
    files: int


def iter_bundle_files(raw_root: Path):
    """Sinh các file cần đưa vào bundle, bỏ qua archive tạm và thư mục ẩn."""
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(raw_root)
        if any(part in EXCLUDED_DIRS or part.startswith(".") for part in relative.parts):
            continue
        yield path, relative


def create_data_bundle(
    raw_root: Path,
    output: Path,
    *,
    archive_prefix: str = "raw",
) -> BundleInfo:
    """Tạo zip và trả về thông tin kiểm tra toàn vẹn."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {raw_root}")

    missing = [name for name in DATASETS if not (raw_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu dataset trong {raw_root}: {', '.join(missing)}. "
            "Chạy `python -m src.data.download --dataset all` trước."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)

    files = 0
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as zf:
        for path, relative in iter_bundle_files(raw_root):
            arcname = str(Path(archive_prefix) / relative).replace("\\", "/")
            zf.write(path, arcname=arcname)
            files += 1

    partial.replace(output)
    info = BundleInfo(
        path=output,
        sha256=sha256_file(output),
        bytes=output.stat().st_size,
        files=files,
    )
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{info.sha256}  {output.name}\n", encoding="utf-8")
    return info


def extract_data_bundle(bundle: Path, dest: Path, *, force: bool = False) -> Path:
    """Giải nén bundle vào `dest` (mặc định nên là thư mục data root)."""
    if dest.exists() and any(dest.iterdir()) and not force:
        return dest
    safe_extract(bundle, dest)
    return dest
