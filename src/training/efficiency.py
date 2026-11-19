"""Đo params, FLOPs, latency và VRAM cho một checkpoint."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.models import build_model, count_parameters
from src.training.checkpoint import load_best


@dataclass
class EfficiencyResult:
    run_id: str
    model: str
    params: int
    flops: int
    gflops: float
    latency_ms_batch1: float
    latency_ms_batch32: float
    throughput_images_per_sec: float
    vram_mb: float | None
    device: str


def build_model_from_config(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    name = str(model_cfg.pop("name"))
    num_classes = int(model_cfg.pop("num_classes", 5))
    model_cfg["pretrained"] = False
    return build_model(name, num_classes=num_classes, **model_cfg)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> float:
    sample = torch.randn(batch_size, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        _sync(device)
    return (time.perf_counter() - start) / repeats * 1000


def measure_model(
    model: torch.nn.Module,
    *,
    device: torch.device | None = None,
    run_id: str = "model",
    model_name: str = "unknown",
) -> EfficiencyResult:
    target = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    params = count_parameters(model)

    from torch.utils.flop_counter import FlopCounterMode

    sample = torch.randn(1, 3, 224, 224, device=target)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(sample)
    flops = int(counter.get_total_flops())

    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    latency1 = _latency(model, target, 1)
    latency32 = _latency(model, target, 32)
    vram = (
        torch.cuda.max_memory_allocated(target) / 1024**2
        if target.type == "cuda"
        else None
    )
    return EfficiencyResult(
        run_id=run_id,
        model=model_name,
        params=params,
        flops=flops,
        gflops=flops / 1e9,
        latency_ms_batch1=latency1,
        latency_ms_batch32=latency32,
        throughput_images_per_sec=32000 / latency32 if latency32 else 0.0,
        vram_mb=vram,
        device=str(target),
    )


def measure_run(run_dir: Path, *, device: torch.device | None = None) -> EfficiencyResult:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run.get("config") or {}
    model = build_model_from_config(config)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "checkpoints" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Không có checkpoint trong {run_dir}")
    load_best(checkpoint, model, map_location="cpu")
    model_name = str((config.get("model") or {}).get("name", "unknown"))
    return measure_model(
        model,
        device=device,
        run_id=str(run.get("run_id", run_dir.name)),
        model_name=model_name,
    )


def to_dict(result: EfficiencyResult) -> dict[str, Any]:
    return asdict(result)
