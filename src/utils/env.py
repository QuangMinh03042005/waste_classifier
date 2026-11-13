"""Thu thập metadata môi trường cho mỗi lần chạy thực nghiệm."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "pillow",
    "pyyaml",
    "requests",
    "tqdm",
)


def _git(args: list[str], project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def git_info(project_root: Path = PROJECT_ROOT) -> dict[str, str | bool | None]:
    """Commit, branch và trạng thái working tree."""
    status = _git(["status", "--porcelain"], project_root)
    return {
        "commit": _git(["rev-parse", "HEAD"], project_root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], project_root),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Phiên bản các thư viện chính, None nếu chưa cài."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_info() -> dict[str, object]:
    """Thông tin torch, CUDA và GPU (nếu có)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        return {"installed": False, "error": repr(exc)}

    info: dict[str, object] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": f"{props.major}.{props.minor}",
                "device_count": torch.cuda.device_count(),
                "vram_gb": round(props.total_memory / 1024**3, 2),
            }
        )
    return info


def environment_info(project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    """Toàn bộ metadata môi trường, có thể ghi JSON."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "git": git_info(project_root),
        "packages": package_versions(),
        "torch": torch_info(),
    }


def write_environment(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Ghi `environment.json` vào thư mục run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(environment_info(project_root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
