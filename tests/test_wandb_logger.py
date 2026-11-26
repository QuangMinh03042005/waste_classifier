"""Test wrapper W&B."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from src.training.wandb_logger import WandbLogger


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.finished = False

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logged.append({"payload": payload, "step": step})

    def finish(self) -> None:
        self.finished = True


def test_disabled_logger_is_noop() -> None:
    logger = WandbLogger({}, run_id="demo", run_config={})
    assert logger.enabled is False
    logger.log_epoch(1, {}, {}, learning_rate=0.1)
    logger.log_summary({}, None, {})
    logger.finish()


def test_enabled_logger_logs(monkeypatch) -> None:
    run = _FakeRun()
    calls = {}

    def init(**kwargs):
        calls.update(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = WandbLogger(
        {"enabled": True, "project": "waste-classifier", "mode": "offline"},
        run_id="demo",
        run_config={"seed": 42},
    )
    assert logger.enabled is True
    assert calls["project"] == "waste-classifier"
    assert calls["mode"] == "offline"

    logger.log_epoch(
        1,
        {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.4},
        {"loss": 0.9, "accuracy": 0.6, "macro_f1": 0.5},
        learning_rate=0.001,
    )
    payload = run.logged[0]["payload"]
    assert payload["val/macro_f1"] == 0.5
    assert run.logged[0]["step"] == 1

    logger.log_summary({"epoch": 1, "macro_f1": 0.5}, {"macro_f1": 0.4}, {"params": 100})
    logger.finish()
    assert run.finished is True
