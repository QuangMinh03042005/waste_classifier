"""Test script sinh bảng layer-by-layer."""

from __future__ import annotations

from scripts.model_summary import collect_stats, render_markdown
from src.models import build_custom_cnn


def test_collect_stats_totals() -> None:
    stats = collect_stats(build_custom_cnn(), image_size=224)
    assert sum(item.params for item in stats) == 1_174_501
    # 2,59 GMAC ≈ 5,17 GFLOPs
    assert 2_550 < sum(item.macs for item in stats) / 1e6 < 2_620


def test_collect_stats_stage_shapes() -> None:
    stats = {item.name: item.output for item in collect_stats(build_custom_cnn())}
    assert stats["stem.0"] == "32×224×224"
    assert stats["stages.0.1"] == "32×112×112"
    assert stats["stages.1.2"] == "64×56×56"
    assert stats["stages.2.2"] == "128×28×28"
    assert stats["stages.3.2"] == "256×14×14"
    assert stats["classifier.1"] == "5"


def test_render_markdown_contains_totals() -> None:
    content = render_markdown(
        collect_stats(build_custom_cnn()),
        model_name="custom_cnn",
        image_size=224,
    )
    assert "| `stages.0` |" in content
    assert "**1,174,501**" in content
    assert "5.173" in content
