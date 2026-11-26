"""Test entrypoint RQ2 (wrapper mỏng quanh training engine)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.rq2 import train


def test_parser_accepts_rq2_arguments() -> None:
    args = train.build_parser().parse_args(
        [
            "--config",
            "configs/rq2/rq2_mobilenetv3s_full.yaml",
            "--plan",
            "plans/rq2/augmentation_ablation.yaml",
            "--seed",
            "42",
            "--epochs",
            "1",
            "--limit-batches",
            "1",
            "--resume",
        ]
    )

    assert args.config == Path("configs/rq2/rq2_mobilenetv3s_full.yaml")
    assert args.plan == Path("plans/rq2/augmentation_ablation.yaml")
    assert args.seed == 42
    assert args.epochs == 1
    assert args.limit_batches == 1
    assert args.resume is True
    assert args.allow_unplanned is False


def test_parser_requires_config() -> None:
    with pytest.raises(SystemExit):
        train.build_parser().parse_args([])
