"""Test cho module tải dữ liệu (không gọi mạng)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.data import download as dl
from src.data.manifest import DATASETS, DatasetSpec


def make_spec(**overrides) -> DatasetSpec:
    defaults = dict(
        name="fake",
        url="https://example.invalid/fake.zip",
        archive_name="fake.zip",
        root_in_archive="root",
        expected_counts={"A": 2, "B": 1},
        license="test",
        citation="test",
        homepage="https://example.invalid",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


def make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return path


def test_manifest_datasets_are_valid() -> None:
    assert set(DATASETS) == {"realwaste", "trashnet"}
    for spec in DATASETS.values():
        assert spec.url.startswith("https://")
        assert spec.expected_total == sum(spec.expected_counts.values())
        assert len(spec.expected_counts) >= 5


def test_realwaste_manifest_counts() -> None:
    spec = DATASETS["realwaste"]
    assert spec.expected_total == 4752
    assert spec.expected_counts["Plastic"] == 921


def test_trashnet_manifest_counts() -> None:
    spec = DATASETS["trashnet"]
    assert spec.expected_total == 2527
    assert spec.expected_counts["trash"] == 137


def test_safe_extract_rejects_zip_slip(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "evil.zip", {"../evil.txt": b"x"})
    with pytest.raises(RuntimeError, match="không an toàn"):
        dl.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_writes_files(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "ok.zip", {"root/A/a.jpg": b"jpg"})
    dl.safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "root" / "A" / "a.jpg").read_bytes() == b"jpg"


def test_find_dataset_root_exact(tmp_path: Path) -> None:
    spec = make_spec()
    (tmp_path / "root" / "A").mkdir(parents=True)
    (tmp_path / "root" / "B").mkdir()
    assert dl.find_dataset_root(tmp_path, spec) == tmp_path / "root"


def test_find_dataset_root_fallback(tmp_path: Path) -> None:
    spec = make_spec(root_in_archive="missing")
    nested = tmp_path / "top" / "sub"
    (nested / "A").mkdir(parents=True)
    (nested / "B").mkdir()
    assert dl.find_dataset_root(tmp_path, spec) == nested


def test_count_images(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "A").mkdir(parents=True)
    (root / "A" / "1.jpg").write_bytes(b"x")
    (root / "A" / "2.PNG").write_bytes(b"x")
    (root / "A" / "notes.txt").write_text("ignore")
    (root / "B").mkdir()
    (root / "B" / "1.jpeg").write_bytes(b"x")
    (root / "empty").mkdir()
    assert dl.count_images(root) == {"A": 2, "B": 1}


def test_compare_counts() -> None:
    assert dl.compare_counts({"a": 2, "B": 1}, {"A": 2, "b": 1}) == []

    problems = dl.compare_counts({"A": 1, "Extra": 3}, {"A": 2, "B": 1})
    text = "\n".join(problems)
    assert "mong đợi 2" in text
    assert "thiếu lớp 'B'" in text
    assert "lớp lạ 'Extra'" in text


def test_verify_sha256(tmp_path: Path) -> None:
    file = tmp_path / "x.bin"
    file.write_bytes(b"hello")
    actual = dl.sha256_file(file)
    assert dl.verify_sha256(file, actual) == actual
    with pytest.raises(RuntimeError, match="SHA256"):
        dl.verify_sha256(file, "0" * 64)


def test_is_dataset_ready(tmp_path: Path) -> None:
    spec = make_spec()
    root = tmp_path / "fake" / "root"
    for name, count in spec.expected_counts.items():
        (root / name).mkdir(parents=True)
        for i in range(count):
            (root / name / f"{i}.jpg").write_bytes(b"x")

    assert not dl.is_dataset_ready(spec, tmp_path)

    manifest = {"root_name": "root", "archive_sha256": "abc"}
    (tmp_path / "fake" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert dl.is_dataset_ready(spec, tmp_path)


def test_download_dataset_skips_when_ready(tmp_path: Path, monkeypatch) -> None:
    spec = make_spec()
    root = tmp_path / "fake" / "root"
    for name, count in spec.expected_counts.items():
        (root / name).mkdir(parents=True)
        for i in range(count):
            (root / name / f"{i}.jpg").write_bytes(b"x")
    (tmp_path / "fake" / "MANIFEST.json").write_text(
        json.dumps({"root_name": "root"}), encoding="utf-8"
    )

    def fail_download(*args, **kwargs):
        raise AssertionError("không được tải lại khi dữ liệu đã đủ")

    monkeypatch.setattr(dl, "download_file", fail_download)
    assert dl.download_dataset(spec, tmp_path) == root
