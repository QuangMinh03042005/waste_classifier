"""Logging tuỳ chọn lên Weights & Biases."""

from __future__ import annotations

import os
from typing import Any


class WandbLogger:
    """Bọc wandb.init/log/finish; no-op khi disabled."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        run_id: str,
        run_config: dict[str, Any],
    ) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.run = None
        if not self.enabled:
            return
        import wandb

        tags = config.get("tags") or []
        mode = os.environ.get("WANDB_MODE") or str(config.get("mode", "online"))
        self.run = wandb.init(
            project=str(config.get("project", "waste-classifier")),
            entity=config.get("entity") or None,
            name=str(config.get("name") or run_id),
            config=run_config,
            tags=list(tags),
            mode=mode,
            reinit=True,
        )

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
        *,
        learning_rate: float,
    ) -> None:
        if self.run is None:
            return
        self.run.log(
            {
                "epoch": epoch,
                "train/loss": float(train_metrics.get("loss", 0.0)),
                "train/accuracy": float(train_metrics.get("accuracy", 0.0)),
                "train/macro_f1": float(train_metrics.get("macro_f1", 0.0)),
                "val/loss": float(val_metrics.get("loss", 0.0)),
                "val/accuracy": float(val_metrics.get("accuracy", 0.0)),
                "val/macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                "learning_rate": float(learning_rate),
            },
            step=epoch,
        )

    def log_summary(
        self,
        best_val: dict[str, Any],
        test: dict[str, Any] | None,
        efficiency: dict[str, Any],
    ) -> None:
        if self.run is None:
            return
        payload: dict[str, Any] = {
            "best_val/epoch": best_val.get("epoch"),
            "best_val/macro_f1": best_val.get("macro_f1"),
            "params": efficiency.get("params"),
            "train_time_sec": efficiency.get("train_time_sec"),
        }
        if test:
            for key, value in test.items():
                if isinstance(value, (int, float)):
                    payload[f"test/{key}"] = value
        self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
