"""Tải và giải nén RealWaste và TrashNet.

Cách dùng:
    uv run python -m src.data.download --dataset all
    uv run python -m src.data.download --dataset realwaste
    uv run python -m src.data.download --dataset trashnet --force

Dữ liệu được giữ nguyên cấu trúc gốc của nhà cung cấp:
    data/raw/realwaste/RealWaste/<Class>/*.jpg
    data/raw/trashnet/dataset-resized/<class>/*.jpg

Lần tải đầu tiên ghi SHA256 vào MANIFEST.json (trust on first use); các lần
tải lại sẽ kiểm tra hash nếu manifest đã tồn tại.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

from src.data.manifest import DATASETS, DatasetSpec

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHUNK_SIZE = 1 << 20
USER_AGENT = "waste-classifier/0.1 (dataset downloader)"


def sha256_file(path: Path) -> str:
    """Tính SHA256 của một file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str | None) -> str:
    """Kiểm tra SHA256 nếu có hash mong đợi; trả về hash thực tế."""
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"SHA256 không khớp cho {path.name}:\n"
            f"  mong đợi: {expected}\n"
            f"  thực tế:  {actual}"
        )
    return actual


def download_file(url: str, dest: Path, *, retries: int = 3) -> int:
    """Tải file với progress bar, retry và ghi tạm trước khi đổi tên."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(10, 60),
                headers={"User-Agent": USER_AGENT},
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length") or 0)
                with (
                    partial.open("wb") as fh,
                    tqdm(
                        total=total or None,
                        unit="B",
                        unit_scale=True,
                        desc=dest.name,
                        leave=False,
                    ) as bar,
                ):
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            partial.replace(dest)
            return dest.stat().st_size
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2**attempt)

    raise RuntimeError(f"Không tải được {url} sau {retries} lần: {last_error}")


def safe_extract(archive: Path, dest: Path) -> None:
    """Giải nén zip, chặn path traversal (zip-slip)."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Archive bị hỏng tại: {corrupt}")

        members = []
        for member in zf.infolist():
            normalized = member.filename.replace("\\", "/")
            member_path = PurePosixPath(normalized)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Đường dẫn không an toàn trong zip: {member.filename}")
            target = (dest / member_path).resolve()
            if not target.is_relative_to(dest):
                raise RuntimeError(f"Đường dẫn không an toàn trong zip: {member.filename}")
            members.append(member)

        zf.extractall(dest)


def find_dataset_root(extract_dir: Path, spec: DatasetSpec) -> Path:
    """Tìm thư mục gốc của dataset sau khi giải nén."""
    candidate = extract_dir / spec.root_in_archive
    if candidate.is_dir():
        return candidate

    expected = {name.lower() for name in spec.expected_counts}
    for path in sorted(extract_dir.rglob("*")):
        if not path.is_dir():
            continue
        subdirs = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        if expected <= subdirs:
            return path

    raise RuntimeError(
        f"Không tìm thấy thư mục dataset '{spec.root_in_archive}' trong {extract_dir}"
    )


def count_images(root: Path) -> dict[str, int]:
    """Đếm số ảnh trong từng thư mục lớp con trực tiếp."""
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        total = sum(
            1 for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
        )
        if total:
            counts[class_dir.name] = total
    return counts


def compare_counts(actual: dict[str, int], expected: dict[str, int]) -> list[str]:
    """So sánh số ảnh thực tế với manifest, trả về danh sách lỗi."""
    actual_lower = {name.lower(): (name, count) for name, count in actual.items()}
    expected_lower = {name.lower(): (name, count) for name, count in expected.items()}
    problems: list[str] = []

    for key, (name, want) in expected_lower.items():
        if key not in actual_lower:
            problems.append(f"thiếu lớp '{name}'")
            continue
        _, got = actual_lower[key]
        if got != want:
            problems.append(f"lớp '{name}': mong đợi {want}, thực tế {got}")

    for key, (name, got) in actual_lower.items():
        if key not in expected_lower:
            problems.append(f"lớp lạ '{name}': {got} ảnh")

    return problems


def read_manifest(path: Path) -> dict | None:
    """Đọc MANIFEST.json nếu tồn tại và hợp lệ."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataset_root(spec: DatasetSpec, data_root: Path, manifest: dict | None) -> Path | None:
    """Suy ra thư mục gốc của dataset từ manifest đã lưu."""
    if not manifest or not manifest.get("root_name"):
        return None
    candidate = data_root / spec.name / str(manifest["root_name"])
    return candidate if candidate.is_dir() else None


def is_dataset_ready(spec: DatasetSpec, data_root: Path) -> bool:
    """True nếu dataset đã tải đủ và số ảnh khớp manifest."""
    manifest = read_manifest(data_root / spec.name / "MANIFEST.json")
    root = dataset_root(spec, data_root, manifest)
    if root is None:
        return False
    return not compare_counts(count_images(root), spec.expected_counts)


def download_dataset(
    spec: DatasetSpec,
    data_root: Path = Path("data/raw"),
    *,
    force: bool = False,
    keep_archive: bool = False,
    skip_validation: bool = False,
) -> Path:
    """Tải, giải nén và kiểm tra một dataset. Trả về thư mục gốc dữ liệu."""
    data_root = data_root.resolve()
    manifest_path = data_root / spec.name / "MANIFEST.json"
    previous = read_manifest(manifest_path)

    existing_root = dataset_root(spec, data_root, previous)
    if not force and is_dataset_ready(spec, data_root) and existing_root is not None:
        print(f"[{spec.name}] đã có sẵn: {existing_root}")
        return existing_root

    archive = data_root / "_downloads" / spec.archive_name
    if force:
        archive.unlink(missing_ok=True)

    if not archive.exists() or archive.stat().st_size == 0:
        print(f"[{spec.name}] đang tải từ {spec.url}")
        download_file(spec.url, archive)
    else:
        print(f"[{spec.name}] dùng archive có sẵn: {archive}")

    expected_sha = previous.get("archive_sha256") if previous else None
    archive_sha = verify_sha256(archive, expected_sha)

    tmp_dir = data_root / ".tmp" / spec.name
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    try:
        print(f"[{spec.name}] đang giải nén...")
        safe_extract(archive, tmp_dir)
        extracted_root = find_dataset_root(tmp_dir, spec)

        counts = count_images(extracted_root)
        problems = compare_counts(counts, spec.expected_counts)
        if problems and not skip_validation:
            detail = "\n  - ".join(problems)
            raise RuntimeError(f"Số lượng ảnh không khớp manifest:\n  - {detail}")

        final_parent = data_root / spec.name
        if final_parent.exists():
            shutil.rmtree(final_parent)
        final_parent.mkdir(parents=True)

        final_root = final_parent / extracted_root.name
        shutil.move(str(extracted_root), str(final_root))

        manifest = {
            "dataset": spec.name,
            "root_name": final_root.name,
            "source_url": spec.url,
            "homepage": spec.homepage,
            "license": spec.license,
            "citation": spec.citation,
            "archive_sha256": archive_sha,
            "archive_bytes": archive.stat().st_size,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "total_images": sum(counts.values()),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not keep_archive:
        archive.unlink(missing_ok=True)

    print(
        f"[{spec.name}] xong: {sum(counts.values())} ảnh / {len(counts)} lớp "
        f"tại {final_root}"
    )
    return final_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tải dataset RealWaste và TrashNet.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        default="all",
        help="Dataset cần tải (mặc định: all).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Thư mục gốc chứa dữ liệu thô (mặc định: data/raw).",
    )
    parser.add_argument("--force", action="store_true", help="Tải lại dù đã có dữ liệu.")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Giữ file zip sau khi giải nén.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Bỏ qua kiểm tra số lượng ảnh so với manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    specs = list(DATASETS.values()) if args.dataset == "all" else [DATASETS[args.dataset]]

    for spec in specs:
        download_dataset(
            spec,
            args.data_root,
            force=args.force,
            keep_archive=args.keep_archive,
            skip_validation=args.skip_validation,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
