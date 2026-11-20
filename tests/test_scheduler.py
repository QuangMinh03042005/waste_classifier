"""Test warmup scheduler."""

from __future__ import annotations

import torch
from torch import nn

from src.training.runner import _make_scheduler


def test_warmup_then_cosine() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = _make_scheduler(optimizer, epochs=10, lr=1e-3, warmup_epochs=3)

    learning_rates = []
    for _ in range(10):
        optimizer.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert learning_rates[0] < learning_rates[2]
    assert learning_rates[-1] < learning_rates[3]
