"""Fit temperature scaling và threshold trên validation cho demo.

Cách dùng:
    uv run python scripts/fit_calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import WasteDataset
from src.data.labels import CANONICAL_CLASSES
from src.data.transforms import build_transforms
from src.inference.calibration import (
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
)
from src.training.checkpoint import load_best
from src.training.efficiency import build_model_from_config
from src.utils.paths import data_root as default_data_root
from src.utils.paths import experiments_root as default_experiments_root
from src.utils.paths import project_root, splits_dir

RUNS = {
    "rq1_customcnn_p20_s42": "Custom CNN",
    "rq1_mobilenetv3s_full_p20_s42": "MobileNet pretrained",
    "rq1_mobilenetv3s_scratch_p20_s42": "MobileNet scratch",
}


def _resolve_data_root(config: dict[str, Any]) -> Path:
    configured = (config.get("data") or {}).get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else project_root() / path
    return default_data_root()


def collect_logits(
    run_dir: Path,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Chạy validation và trả logits/labels, không đụng test."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run.get("config") or {}
    data_cfg = config["data"]
    class_names = list(data_cfg.get("class_names") or CANONICAL_CLASSES)
    image_size = int(data_cfg.get("image_size", 224))
    split_csv = splits_dir() / str(data_cfg.get("splits_version", "v1")) / "split.csv"
    dataset = WasteDataset(
        data_root=_resolve_data_root(config),
        split_csv=split_csv,
        split="val",
        class_names=class_names,
        transform=build_transforms(train=False, image_size=image_size),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model_from_config(config)
    load_best(run_dir / "checkpoints" / "best.pt", model, map_location=device)
    model.to(device).eval()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            logits_list.append(logits.float().cpu())
            labels_list.append(labels.cpu())
    return (
        torch.cat(logits_list).numpy(),
        torch.cat(labels_list).numpy(),
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Fit calibration trên validation.")
    parser.add_argument("--experiments-root", type=Path, default=default_experiments_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/demo/calibration.yaml"),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = args.experiments_root.resolve()
    calibration: dict[str, dict[str, float]] = {}
    for run_id in RUNS:
        run_dir = root / "rq1" / run_id
        if not run_dir.is_dir():
            print(f"Thiếu run: {run_id}")
            continue
        logits, labels = collect_logits(run_dir, device=device)
        temperature = fit_temperature(logits, labels)
        scaled = torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()
        threshold = choose_threshold(scaled, labels)
        ece_before = expected_calibration_error(
            torch.softmax(torch.as_tensor(logits), dim=1).numpy(), labels
        )
        ece_after = expected_calibration_error(scaled, labels)
        calibration[run_id] = {
            "temperature": round(temperature, 4),
            "threshold": round(threshold, 4),
        }
        print(
            f"{run_id}: n={len(labels)} T={temperature:.3f} "
            f"threshold={threshold:.3f} ECE trước={ece_before:.4f} "
            f"sau={ece_after:.4f}"
        )

    if not calibration:
        print("Không có model nào để fit.")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(calibration, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Đã ghi {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
