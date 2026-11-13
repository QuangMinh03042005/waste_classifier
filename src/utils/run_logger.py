"""Ghi log cho một lần chạy thực nghiệm.

Mỗi run có một thư mục riêng, không ghi đè. Log gồm:

- `run.json`       metadata + config + trạng thái (ghi atomic)
- `history.jsonl`  một dòng mỗi epoch/phase
- `metrics.json`   kết quả cuối (best validation + test)
- `error.json`     chỉ có khi run thất bại
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils import env as env_utils
from src.utils.paths import experiments_root as default_experiments_root

SCHEMA_VERSION = "1.0"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RQ_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RunExistsError(FileExistsError):
    """Thư mục run đã tồn tại và không cho phép ghi đè."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Ghi JSON qua file tạm rồi đổi tên, tránh file hỏng khi bị ngắt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append một dòng JSONL và flush ngay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        fh.flush()


def read_history(path: Path) -> list[dict[str, Any]]:
    """Đọc lại toàn bộ history.jsonl."""
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class RunLogger:
    """Quản lý thư mục và file log của một run."""

    def __init__(
        self,
        rq: str,
        run_id: str,
        config: dict[str, Any],
        *,
        seed: int | None = None,
        experiments_root: Path | None = None,
        resume: bool = False,
        from_epoch: int | None = None,
        overwrite: bool = False,
    ) -> None:
        if not _RQ_RE.match(rq):
            raise ValueError(f"rq không hợp lệ: {rq!r}")
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"run_id không hợp lệ: {run_id!r}")
        if overwrite and resume:
            raise ValueError("Không thể vừa overwrite vừa resume.")

        resolved_seed = seed if seed is not None else config.get("seed")
        if resolved_seed is None:
            raise ValueError("Thiếu seed (truyền seed= hoặc đặt config['seed']).")

        base = experiments_root if experiments_root is not None else default_experiments_root()
        self.root = base / rq / run_id
        self.run_json = self.root / "run.json"
        self.history = self.root / "history.jsonl"
        self.metrics = self.root / "metrics.json"
        self.environment = self.root / "environment.json"
        self.error = self.root / "error.json"
        self.checkpoints = self.root / "checkpoints"
        self.predictions = self.root / "predictions"
        self.figures = self.root / "figures"

        self.rq = rq
        self.run_id = run_id
        self.seed = int(resolved_seed)
        self.experiment = config.get("experiment") or re.sub(r"_s\d+$", "", run_id)
        self._started_monotonic = time.monotonic()

        if resume:
            self._init_resume(from_epoch)
        else:
            if self.root.exists() and not overwrite:
                raise RunExistsError(
                    f"Run đã tồn tại: {self.root}. Dùng run_id mới hoặc resume=True."
                )
            self._init_new(rq, run_id, config)

    def _init_new(self, rq: str, run_id: str, config: dict[str, Any]) -> None:
        for directory in (self.root, self.checkpoints, self.predictions, self.figures):
            directory.mkdir(parents=True, exist_ok=True)

        (self.root / "config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        env_utils.write_environment(self.environment)

        self._run: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "rq": rq,
            "experiment": self.experiment,
            "seed": self.seed,
            "status": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "duration_sec": None,
            "git": env_utils.git_info(),
            "config": config,
            "environment": "environment.json",
            "checkpoint": {},
            "resume": {"count": 0, "from_epoch": None},
        }
        self._save()

    def _init_resume(self, from_epoch: int | None) -> None:
        if not self.run_json.is_file():
            raise FileNotFoundError(f"Không tìm thấy run để resume: {self.run_json}")
        self._run = json.loads(self.run_json.read_text(encoding="utf-8"))
        self.experiment = self._run.get("experiment", self.experiment)
        resume = self._run.setdefault("resume", {"count": 0, "from_epoch": None})
        resume["count"] = int(resume.get("count", 0)) + 1
        resume["from_epoch"] = from_epoch
        self._run["status"] = "running"
        self._run["finished_at"] = None
        self._run["updated_at"] = utc_now()
        self._save()

    def _save(self) -> None:
        self._run["updated_at"] = utc_now()
        write_json_atomic(self.run_json, self._run)

    def update(self, *, persist: bool = True, **fields: Any) -> None:
        """Cập nhật run.json với thông tin bổ sung."""
        self._run.update(fields)
        if persist:
            self._save()

    def log_epoch(
        self,
        epoch: int,
        phase: str,
        metrics: dict[str, Any],
        *,
        duration_sec: float | None = None,
    ) -> None:
        """Ghi metric của một epoch/phase vào history.jsonl."""
        record: dict[str, Any] = {**metrics}
        record["epoch"] = int(epoch)
        record["phase"] = str(phase)
        if duration_sec is not None:
            record["duration_sec"] = float(duration_sec)
        record["at"] = utc_now()
        append_jsonl(self.history, record)
        self._run["last_epoch"] = int(epoch)

    def log_event(self, event: str, **fields: Any) -> None:
        """Ghi sự kiện (checkpoint, early_stop, note...)."""
        record: dict[str, Any] = {**fields}
        record["event"] = str(event)
        record["at"] = utc_now()
        append_jsonl(self.history, record)

    def log_checkpoint(
        self,
        kind: str,
        path: Path,
        *,
        epoch: int,
        metric: float | None = None,
    ) -> None:
        """Ghi thông tin checkpoint; chỉ best mới lưu run.json ngay."""
        try:
            relative = str(path.relative_to(self.root))
        except ValueError:
            relative = str(path)
        self._run.setdefault("checkpoint", {})[kind] = {
            "path": relative,
            "epoch": int(epoch),
            "metric": None if metric is None else float(metric),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        if kind == "best":
            self._save()

    def finish(self, metrics: dict[str, Any], *, status: str = "completed") -> Path:
        """Ghi metrics.json và đóng run."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "class_names": metrics.get("class_names"),
            **metrics,
        }
        write_json_atomic(self.metrics, payload)

        best = metrics.get("best_val") or {}
        test = metrics.get("test") or {}
        self._run["status"] = status
        self._run["finished_at"] = utc_now()
        self._run["duration_sec"] = round(time.monotonic() - self._started_monotonic, 3)
        self._run["best_epoch"] = best.get("epoch")
        self._run["result"] = {
            "best_val_macro_f1": best.get("macro_f1"),
            "test_macro_f1": test.get("macro_f1"),
        }
        self._save()
        return self.metrics

    def fail(self, exc: BaseException) -> Path:
        """Ghi error.json và đánh dấu run thất bại."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
            "at": utc_now(),
        }
        write_json_atomic(self.error, payload)
        self._run["status"] = "failed"
        self._run["finished_at"] = utc_now()
        self._run["duration_sec"] = round(time.monotonic() - self._started_monotonic, 3)
        self._save()
        return self.error

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None and self._run.get("status") == "running":
            self.fail(exc)
        return False
