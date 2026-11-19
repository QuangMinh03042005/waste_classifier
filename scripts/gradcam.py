"""Sinh Grad-CAM cho các ảnh dự đoán sai của một run.

Cách dùng:
    uv run python scripts/gradcam.py --run experiments/rq1/rq1_customcnn_s42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import CANONICAL_CLASSES  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.training.checkpoint import load_best  # noqa: E402
from src.training.efficiency import build_model_from_config  # noqa: E402
from src.training.gradcam import GradCAM, overlay_cam  # noqa: E402
from src.utils.paths import project_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grad-CAM cho run RQ1.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="CSV predictions; mặc định run/predictions/test.csv",
    )
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--target", choices=["pred", "true"], default="pred")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _resolve_data_root(config: dict) -> Path:
    configured = (config.get("data") or {}).get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else project_root() / path
    from src.utils.paths import data_root

    return data_root()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    run_dir = args.run
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run.get("config") or {}
    class_names = list(
        (config.get("data") or {}).get("class_names") or CANONICAL_CLASSES
    )
    if not class_names:
        print("Config thiếu class_names.")
        return 1
    image_size = int((config.get("data") or {}).get("image_size", 224))
    data_root = _resolve_data_root(config)

    predictions_file = args.predictions or (run_dir / "predictions" / "test.csv")
    if not predictions_file.is_file():
        print(f"Không có predictions: {predictions_file}")
        return 1
    with predictions_file.open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("correct") == "0"]
    if not rows:
        print("Không có ảnh dự đoán sai.")
        return 0
    selected = rows[: args.num]

    checkpoint = args.checkpoint or (run_dir / "checkpoints" / "best.pt")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(config)
    load_best(checkpoint, model, map_location=device)
    model.to(device)

    cam = GradCAM(model, args.layer)
    eval_transform = build_transforms(train=False, image_size=image_size)
    display_transform = transforms.Compose(
        [transforms.Resize(round(image_size * 256 / 224)), transforms.CenterCrop(image_size)]
    )
    output_dir = args.output_dir or (run_dir / "figures" / "gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, row in enumerate(selected, start=1):
            image_path = data_root / row["path"]
            image = Image.open(image_path).convert("RGB")
            tensor = eval_transform(image).unsqueeze(0).to(device)
            target_label = row["pred_label"] if args.target == "pred" else row["true_label"]
            target_index = class_names.index(target_label)
            heatmap, _ = cam.generate(tensor, target_index)
            display = display_transform(image)
            overlay = overlay_cam(display, heatmap)

            figure, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(display)
            axes[0].set_title(f"True: {row['true_label']}")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Pred: {row['pred_label']} | Grad-CAM")
            for axis in axes:
                axis.axis("off")
            figure.tight_layout()
            output = output_dir / f"{index:02d}_{row['true_label']}_to_{row['pred_label']}.png"
            figure.savefig(output, dpi=140)
            plt.close(figure)
            print(f"Đã ghi {output}")
    finally:
        cam.remove()

    print(f"Xong {len(selected)} ảnh tại {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
