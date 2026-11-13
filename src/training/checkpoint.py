"""Lưu và khôi phục checkpoint cho training.

- `last.pt`: đầy đủ state để resume (model, optimizer, scheduler, scaler, RNG).
- `best.pt`: nhẹ, chỉ weights + metadata để inference/demo.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

SCHEMA_VERSION = "1.0"

_META_KEYS = (
    "schema_version",
    "kind",
    "epoch",
    "global_step",
    "best_metric",
    "class_names",
    "config",
    "seed",
    "git_commit",
)


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    path: Path
    epoch: int | None
    best_metric: float | None
    class_names: list[str] | None
    config: dict[str, Any] | None


def capture_rng_state() -> dict[str, Any]:
    """Chụp trạng thái RNG để resume tái lập được."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Khôi phục trạng thái RNG đã lưu."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_last(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    class_names: list[str],
    config: dict[str, Any],
    seed: int,
    scheduler: Any = None,
    scaler: Any = None,
    git_commit: str | None = None,
    rng_state: dict[str, Any] | None = None,
) -> Path:
    """Lưu checkpoint đầy đủ để resume."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "last",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "class_names": list(class_names),
        "config": config,
        "seed": int(seed),
        "git_commit": git_commit,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
    }
    _atomic_save(payload, path)
    return path


def save_best(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    metric: float,
    class_names: list[str],
    config: dict[str, Any],
    seed: int,
    git_commit: str | None = None,
) -> Path:
    """Lưu checkpoint nhẹ để inference; không gồm optimizer."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "best",
        "epoch": int(epoch),
        "best_metric": float(metric),
        "class_names": list(class_names),
        "config": config,
        "seed": int(seed),
        "git_commit": git_commit,
        "model_state": _cpu_state_dict(model),
    }
    _atomic_save(payload, path)
    return path


def _load_raw(path: Path, map_location: Any) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def read_checkpoint_meta(path: Path) -> CheckpointInfo:
    """Đọc metadata mà không cần dựng model."""
    payload = _load_raw(path, "cpu")
    return CheckpointInfo(
        path=path,
        epoch=payload.get("epoch"),
        best_metric=payload.get("best_metric"),
        class_names=payload.get("class_names"),
        config=payload.get("config"),
    )


def load_best(path: Path, model: nn.Module, *, map_location: Any = "cpu") -> CheckpointInfo:
    """Nạp best.pt vào model và trả metadata."""
    payload = _load_raw(path, map_location)
    model.load_state_dict(payload["model_state"])
    return CheckpointInfo(
        path=path,
        epoch=payload.get("epoch"),
        best_metric=payload.get("best_metric"),
        class_names=payload.get("class_names"),
        config=payload.get("config"),
    )


def load_last(
    path: Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> CheckpointInfo:
    """Khôi phục đầy đủ trạng thái training từ last.pt."""
    payload = _load_raw(path, map_location)
    model.load_state_dict(payload["model_state"])

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])

    return CheckpointInfo(
        path=path,
        epoch=payload.get("epoch"),
        best_metric=payload.get("best_metric"),
        class_names=payload.get("class_names"),
        config=payload.get("config"),
    )
