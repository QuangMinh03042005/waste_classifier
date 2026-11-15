"""Đóng gói các run thực nghiệm thành một file zip để tải về.

Zip có cấu trúc `experiments/...` để giải nén ngay tại repo root.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils import env as env_utils

SCHEMA_VERSION = "1.0"
SKIP_DIRS = {".tmp", "__pycache__"}
EXPORT_DIR = "_export"


@dataclass(frozen=True, slots=True)
class ExportInfo:
    path: Path
    sha256: str
    bytes: int
    files: int
    runs: int
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_runs(
    experiments_root: Path,
    *,
    rq: str | None = None,
    experiment: str | None = None,
    include: list[str] | None = None,
    include_incomplete: bool = False,
) -> list[Path]:
    """Chọn các thư mục run theo rq/experiment/run_id và trạng thái."""
    experiments_root = Path(experiments_root)
    wanted = set(include or [])
    selected: list[Path] = []

    for run_file in sorted(experiments_root.glob("*/*/run.json")):
        run = _read_json(run_file)
        if not run:
            continue
        run_rq = run.get("rq") or run_file.parent.parent.name
        if rq and run_rq != rq:
            continue
        if experiment and run.get("experiment") != experiment:
            continue
        if wanted and run.get("run_id", run_file.parent.name) not in wanted:
            continue
        if not include_incomplete and run.get("status") != "completed":
            continue
        selected.append(run_file.parent)
    return selected


def _run_files(run_dir: Path, checkpoints: str):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if any(part in SKIP_DIRS or part.startswith(".") for part in relative.parts):
            continue
        if path.suffix == ".part":
            continue
        if relative.parts[0] == "checkpoints":
            if checkpoints == "none":
                continue
            if checkpoints == "best" and path.name != "best.pt":
                continue
        yield path, relative


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    run = _read_json(run_dir / "run.json") or {}
    metrics = _read_json(run_dir / "metrics.json") or {}
    best = metrics.get("best_val") or {}
    test = metrics.get("test") or {}
    model = run.get("model") or {}
    git = run.get("git") or {}
    return {
        "run_id": run.get("run_id", run_dir.name),
        "rq": run.get("rq", run_dir.parent.name),
        "experiment": run.get("experiment", run_dir.name),
        "seed": run.get("seed"),
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "duration_sec": run.get("duration_sec"),
        "git_commit": git.get("commit"),
        "params": model.get("params"),
        "best_val_macro_f1": best.get("macro_f1"),
        "test_macro_f1": test.get("macro_f1"),
    }


def _readme(run_count: int, checkpoints: str) -> str:
    return (
        "Waste classification experiment export\n"
        "=====================================\n\n"
        f"- Số run: {run_count}\n"
        f"- Checkpoints: {checkpoints}\n\n"
        "Giải nén tại repo root để dữ liệu vào experiments/:\n"
        "  unzip waste_experiments_*.zip -d /path/to/waste_classifier\n\n"
        "Chi tiết run và metric: experiments/_export/manifest.json\n"
        "Tổng hợp bảng: python scripts/collect_results.py\n"
    )


def create_export(
    experiments_root: Path,
    output: Path,
    runs: list[Path],
    *,
    checkpoints: str = "best",
    force: bool = False,
) -> ExportInfo:
    """Tạo zip từ danh sách run."""
    if checkpoints not in {"none", "best", "all"}:
        raise ValueError(f"checkpoints phải là none/best/all, nhận {checkpoints!r}")
    if not runs:
        raise ValueError("Không có run nào để export.")
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"{output} đã tồn tại. Dùng --force để ghi đè.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)

    entries: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0

    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for run_dir in runs:
            relative_run = run_dir.relative_to(experiments_root)
            metadata = _run_metadata(run_dir)
            file_count = 0
            byte_count = 0
            for path, relative in _run_files(run_dir, checkpoints):
                arcname = f"experiments/{relative_run.as_posix()}/{relative.as_posix()}"
                compress = zipfile.ZIP_STORED if path.suffix == ".pt" else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=arcname, compress_type=compress)
                file_count += 1
                byte_count += path.stat().st_size
            metadata.update({"files": file_count, "bytes": byte_count})
            entries.append(metadata)
            total_files += file_count
            total_bytes += byte_count

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": datetime.now(UTC).isoformat(),
            "source_root": str(experiments_root),
            "checkpoints": checkpoints,
            "git": env_utils.git_info(),
            "totals": {
                "runs": len(entries),
                "files": total_files,
                "bytes": total_bytes,
            },
            "runs": entries,
        }
        archive.writestr(
            f"experiments/{EXPORT_DIR}/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr(
            f"experiments/{EXPORT_DIR}/README.txt",
            _readme(len(entries), checkpoints),
        )

    partial.replace(output)
    checksum = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"
    )
    return ExportInfo(
        path=output,
        sha256=checksum,
        bytes=output.stat().st_size,
        files=total_files,
        runs=len(entries),
        manifest=manifest,
    )


def download_on_colab(path: Path) -> bool:
    """Tải file trong Colab bằng google.colab.files; trả False nếu không phải Colab."""
    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        return False
    files.download(str(path))
    return True
