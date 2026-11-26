"""Test chọn ảnh cho so sánh Grad-CAM."""

from __future__ import annotations

import pandas as pd
import torch

from src.training.gradcam_compare import RunBundle, select_cases

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]


def _bundle(tmp_path) -> RunBundle:
    return RunBundle(
        label="Model A",
        run_dir=tmp_path,
        model=torch.nn.Identity(),
        predictions=pd.DataFrame(),
        class_names=CLASS_NAMES,
        data_root=tmp_path,
        image_size=224,
        transform=None,
        display=None,
        device=torch.device("cpu"),
    )


def test_select_cases_one_correct_one_error_per_class(tmp_path) -> None:
    rows = []
    for index, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "path": f"{class_name}_correct.jpg",
                "true_label": class_name,
                "pred::Model A": class_name,
                "confidence::Model A": 0.95,
                "correct::Model A": True,
            }
        )
        rows.append(
            {
                "path": f"{class_name}_wrong.jpg",
                "true_label": class_name,
                "pred::Model A": "plastic" if class_name != "plastic" else "metal",
                "confidence::Model A": 0.90 - index * 0.01,
                "correct::Model A": False,
            }
        )
    merged = pd.DataFrame(rows)
    selected, wrong_cases = select_cases(
        merged, [_bundle(tmp_path)], CLASS_NAMES, source="test", per_class=2
    )

    assert len(selected) == 10
    assert len(wrong_cases) == 3
    for class_name in CLASS_NAMES:
        cases = [case for case in selected if case.true_label == class_name]
        assert len(cases) == 2
        assert any(case.correct["Model A"] for case in cases)
        assert any(not case.correct["Model A"] for case in cases)
