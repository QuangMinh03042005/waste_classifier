"""Test đóng gói và giải nén bundle dữ liệu."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.data.bundle import create_data_bundle, extract_data_bundle
from src.data.download import sha256_file


def make_raw_tree(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    for name in ("realwaste", "trashnet"):
        dataset = raw / name / "root"
        dataset.mkdir(parents=True)
        (dataset / "a.jpg").write_bytes(b"jpg")
    (raw / "_downloads").mkdir()
    (raw / "_downloads" / "archive.zip").write_bytes(b"zip")
    (raw / ".tmp").mkdir()
    (raw / ".tmp" / "partial.bin").write_bytes(b"tmp")
    return raw


def test_create_data_bundle(tmp_path: Path) -> None:
    raw = make_raw_tree(tmp_path)
    output = tmp_path / "dist" / "waste_raw_v1.zip"

    info = create_data_bundle(raw, output)

    assert info.path == output
    assert info.files == 2
    assert info.bytes == output.stat().st_size
    assert info.sha256 == sha256_file(output)
    assert output.with_suffix(".zip.sha256").read_text(encoding="utf-8").startswith(info.sha256)

    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
    assert "raw/realwaste/root/a.jpg" in names
    assert "raw/trashnet/root/a.jpg" in names
    assert not any("_downloads" in name or ".tmp" in name for name in names)


def test_create_data_bundle_requires_both_datasets(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "realwaste" / "root").mkdir(parents=True)
    try:
        create_data_bundle(raw, tmp_path / "bundle.zip")
    except FileNotFoundError as exc:
        assert "trashnet" in str(exc)
    else:
        raise AssertionError("phải báo thiếu dataset")


def test_extract_data_bundle(tmp_path: Path) -> None:
    raw = make_raw_tree(tmp_path)
    output = create_data_bundle(raw, tmp_path / "bundle.zip")
    dest = tmp_path / "content" / "data"

    extract_data_bundle(output.path, dest)

    assert (dest / "raw" / "realwaste" / "root" / "a.jpg").read_bytes() == b"jpg"
    assert (dest / "raw" / "trashnet" / "root" / "a.jpg").exists()
