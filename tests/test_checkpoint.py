"""Test checkpoint last/best."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.training.checkpoint import load_best, load_last, read_checkpoint_meta, save_best, save_last


def make_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(3, 4)


def test_save_and_load_last(tmp_path: Path) -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    path = tmp_path / "checkpoints" / "last.pt"

    info = save_last(
        path,
        model,
        optimizer,
        epoch=3,
        global_step=30,
        best_metric=0.75,
        class_names=["a", "b", "c", "d"],
        config={"seed": 1},
        seed=1,
        git_commit="abc123",
    )
    assert info == path
    assert path.is_file()

    restored = make_model()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    meta = load_last(path, restored, optimizer=restored_optimizer)
    assert meta.epoch == 3
    assert meta.best_metric == 0.75
    assert meta.class_names == ["a", "b", "c", "d"]
    assert torch.equal(model.state_dict()["weight"], restored.state_dict()["weight"])

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "last"
    assert payload["schema_version"] == "1.0"
    assert payload["git_commit"] == "abc123"
    assert payload["optimizer_state"]


def test_save_and_load_best(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "checkpoints" / "best.pt"

    save_best(
        path,
        model,
        epoch=5,
        metric=0.8,
        class_names=["a", "b", "c", "d"],
        config={"seed": 1},
        seed=1,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "best"
    assert "optimizer_state" not in payload

    restored = make_model()
    meta = load_best(path, restored)
    assert meta.epoch == 5
    assert meta.best_metric == 0.8
    assert torch.equal(model.state_dict()["weight"], restored.state_dict()["weight"])


def test_read_metadata(tmp_path: Path) -> None:
    model = make_model()
    path = tmp_path / "best.pt"
    save_best(
        path,
        model,
        epoch=2,
        metric=0.6,
        class_names=["a"],
        config={"seed": 1},
        seed=1,
    )
    info = read_checkpoint_meta(path)
    assert info.epoch == 2
    assert info.config == {"seed": 1}
