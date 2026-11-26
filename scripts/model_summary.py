"""Sinh bảng layer-by-layer (params, output shape, MACs) cho một model.

Cách dùng:
    uv run python scripts/model_summary.py --model custom_cnn
    uv run python scripts/model_summary.py \
        --config configs/rq1/rq1_customcnn.yaml \
        --output docs/RQ1/custom_cnn_layers.md

Số liệu lấy trực tiếp từ forward pass nên bảng luôn khớp code.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.models import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402


@dataclass
class LayerStat:
    """Thống kê một layer lá."""

    name: str
    kind: str
    params: int
    output: str
    macs: int


def _count_macs(module: nn.Module, output: torch.Tensor) -> int:
    """Ước tính MACs của Conv2d/Linear từ output shape."""
    elements = output[0].numel() if output.dim() > 1 else output.numel()
    if isinstance(module, nn.Conv2d):
        kernels = module.kernel_size[0] * module.kernel_size[1]
        return elements * (module.in_channels // module.groups) * kernels
    if isinstance(module, nn.Linear):
        return elements * module.in_features
    return 0


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "-"
    if len(shape) == 4:
        return f"{shape[1]}×{shape[2]}×{shape[3]}"
    if len(shape) == 2:
        return f"{shape[1]}"
    return "×".join(str(dim) for dim in shape)


def collect_stats(model: nn.Module, image_size: int = 224) -> list[LayerStat]:
    """Chạy một forward pass giả và ghi lại shape/MACs từng layer lá."""
    model.eval()
    records: dict[str, tuple[tuple[int, ...], int]] = {}
    handles = []

    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue

        def hook(
            mod: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            layer_name: str = name,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                return
            records[layer_name] = (
                tuple(output.shape),
                _count_macs(mod, output),
            )

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, image_size, image_size))
    finally:
        for handle in handles:
            handle.remove()

    stats: list[LayerStat] = []
    for name, module in model.named_modules():
        if name not in records:
            continue
        shape, macs = records[name]
        stats.append(
            LayerStat(
                name=name,
                kind=type(module).__name__,
                params=sum(p.numel() for p in module.parameters(recurse=False)),
                output=_format_shape(shape),
                macs=macs,
            )
        )
    return stats


def _group_name(layer: str) -> str:
    """Gom layer theo stage: stem, stages.0, ..., pool, classifier."""
    parts = layer.split(".")
    if parts[0] == "stages" and len(parts) > 1:
        return f"stages.{parts[1]}"
    return parts[0]


def render_markdown(
    stats: list[LayerStat],
    *,
    model_name: str,
    image_size: int,
) -> str:
    """Sinh bảng markdown layer-by-layer và bảng tổng hợp theo stage."""
    total_params = sum(item.params for item in stats)
    total_macs = sum(item.macs for item in stats)

    lines = [
        f"# Layer summary: {model_name}",
        "",
        f"Input: 1×3×{image_size}×{image_size}; sinh bởi `scripts/model_summary.py`.",
        "",
        "| Layer | Type | Params | Output | MACs (M) |",
        "|---|---|---:|---|---:|",
    ]
    for item in stats:
        lines.append(
            f"| `{item.name}` | {item.kind} | {item.params:,} | "
            f"{item.output} | {item.macs / 1e6:.3f} |"
        )
    lines.append(f"| **Tổng** | | **{total_params:,}** | | **{total_macs / 1e6:.2f}** |")

    groups: dict[str, list[LayerStat]] = {}
    for item in stats:
        groups.setdefault(_group_name(item.name), []).append(item)

    lines += [
        "",
        "## Tổng hợp theo stage",
        "",
        "| Stage | Layers | Params | MACs (M) |",
        "|---|---|---:|---:|",
    ]
    for group, items in groups.items():
        params = sum(item.params for item in items)
        macs = sum(item.macs for item in items)
        lines.append(f"| `{group}` | {len(items)} | {params:,} | {macs / 1e6:.3f} |")
    lines.append(
        f"| **Tổng** | **{len(stats)}** | **{total_params:,}** | **{total_macs / 1e6:.2f}** |"
    )

    lines += [
        "",
        f"GFLOPs (2 × MACs): **{2 * total_macs / 1e9:.3f}**.",
        "",
    ]
    return "\n".join(lines)


def _build_from_config(path: Path) -> tuple[nn.Module, str]:
    config = load_config(path)
    model_cfg = dict(config.get("model") or {})
    name = str(model_cfg.pop("name"))
    num_classes = int(model_cfg.pop("num_classes", 5))
    model_cfg.pop("pretrained", None)
    model = build_model(name, num_classes=num_classes, **model_cfg)
    return model, name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sinh bảng layer-by-layer của model.")
    parser.add_argument("--model", default="custom_cnn", help="Tên model trong registry.")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML để đọc model.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    if args.config is not None:
        model, model_name = _build_from_config(args.config)
    else:
        model = build_model(args.model)
        model_name = args.model

    stats = collect_stats(model, image_size=args.image_size)
    content = render_markdown(stats, model_name=model_name, image_size=args.image_size)
    print(content)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"Đã ghi {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
