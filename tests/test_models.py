"""Test model registry, Custom CNN và MobileNet transfer utilities."""

from __future__ import annotations

import pytest
import torch

from src.models import (
    CustomCNN,
    build_custom_cnn,
    build_mobilenet_v3_small,
    build_model,
    build_param_groups,
    count_parameters,
    freeze_backbone,
    unfreeze_last_blocks,
)


def test_custom_cnn_forward_and_params() -> None:
    model = build_custom_cnn()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 5)
    params = count_parameters(model)
    assert 1_000_000 < params < 1_400_000


def test_custom_cnn_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="channels"):
        CustomCNN(channels=(32, 64))
    with pytest.raises(ValueError, match="convs_per_stage"):
        CustomCNN(convs_per_stage=(1, 2, 2, 0))
    with pytest.raises(ValueError, match="dropout"):
        CustomCNN(dropout=1.0)


def test_custom_cnn_residual_forward() -> None:
    model = build_custom_cnn(residual=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 5)
    assert model.residual is True
    assert count_parameters(model) > count_parameters(build_custom_cnn(residual=False))


def test_mobilenet_replaces_classifier() -> None:
    model = build_mobilenet_v3_small(pretrained=False, num_classes=5, dropout=0.3)
    assert model.classifier[-1].out_features == 5
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 5)


def test_freeze_and_unfreeze() -> None:
    model = build_mobilenet_v3_small(pretrained=False)
    freeze_backbone(model)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    unfreeze_last_blocks(model, blocks=2)
    blocks = list(model.features.children())
    assert all(not p.requires_grad for p in blocks[0].parameters())
    assert all(p.requires_grad for p in blocks[-1].parameters())
    assert all(p.requires_grad for p in blocks[-2].parameters())


def test_param_groups_cover_trainable_params() -> None:
    model = build_mobilenet_v3_small(pretrained=False)
    freeze_backbone(model)
    unfreeze_last_blocks(model, blocks=2)

    groups = build_param_groups(model, base_lr=1e-3, weight_decay=1e-4)
    assert len(groups) == 2
    lrs = {round(group["lr"], 8) for group in groups}
    assert lrs == {1e-3, 1e-4}

    params = [parameter for group in groups for parameter in group["params"]]
    assert sum(parameter.numel() for parameter in params) == count_parameters(
        model, trainable_only=True
    )
    assert len({id(parameter) for parameter in params}) == len(params)


def test_scratch_param_groups_use_same_lr() -> None:
    model = build_mobilenet_v3_small(pretrained=False)
    groups = build_param_groups(
        model,
        base_lr=1e-3,
        weight_decay=1e-4,
        backbone_lr_scale=1.0,
    )
    assert len(groups) == 2
    assert all(round(group["lr"], 8) == 1e-3 for group in groups)


def test_factory() -> None:
    assert isinstance(build_model("custom_cnn"), CustomCNN)
    with pytest.raises(ValueError, match="registry"):
        build_model("unknown")
