"""Training engine dùng chung cho các RQ.

Entrypoint trong `research/rqX/train.py` gọi `run_training`, truyền config và plan.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import WasteDataset
from src.data.labels import CANONICAL_CLASSES
from src.data.transforms import build_transforms
from src.models import (
    build_model,
    build_param_groups,
    count_parameters,
    freeze_backbone,
    unfreeze_last_blocks,
)
from src.training.checkpoint import load_best, load_last, save_best, save_last
from src.training.metrics import (
    classification_metrics,
    efficiency_metrics,
    write_predictions,
)
from src.training.run import start_run
from src.utils import env as env_utils
from src.utils.config import load_config
from src.utils.paths import data_root as default_data_root
from src.utils.paths import project_root, splits_dir
from src.utils.run_logger import RunLogger


@dataclass
class TrainingResult:
    run_dir: Path
    best_val: dict[str, Any]
    test: dict[str, Any] | None
    history: list[dict[str, Any]] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _resolve_data_root(config: dict[str, Any]) -> Path:
    configured = config["data"].get("root")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (project_root() / path)
    return default_data_root()


def _resolve_split_csv(config: dict[str, Any]) -> Path:
    configured = config["data"].get("split_csv")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (project_root() / path)
    version = str(config["data"].get("splits_version", "v1"))
    return splits_dir() / version / "split.csv"


def _make_loader(
    dataset: WasteDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    groups = build_param_groups(
        model, base_lr=lr, weight_decay=weight_decay, backbone_lr_scale=0.1
    )
    if not groups:
        raise RuntimeError("Model không có tham số trainable.")
    return torch.optim.AdamW(groups)


def _make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, lr: float
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01
    )


def _class_weights(dataset: WasteDataset, *, clip: tuple[float, float]) -> torch.Tensor:
    counts = dataset.class_counts
    total = sum(counts.values())
    classes = len(counts)
    weights = [total / (classes * max(count, 1)) for count in counts.values()]
    weights = [min(max(value, clip[0]), clip[1]) for value in weights]
    return torch.tensor(weights, dtype=torch.float32)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_names: Sequence[str],
    *,
    use_amp: bool,
    max_steps: int | None,
    grad_clip: float | None,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

        if max_steps is not None and step >= max_steps:
            break

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    *,
    use_amp: bool,
    max_steps: int | None,
    collect_probabilities: bool = False,
) -> tuple[dict[str, Any], list[int], list[int], list[list[float]]]:
    model.eval()
    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if collect_probabilities:
            probabilities.extend(
                torch.softmax(logits.float(), dim=1).detach().cpu().tolist()
            )

        if max_steps is not None and step >= max_steps:
            break

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics["loss"] = total_loss / max(seen, 1)
    return metrics, y_true, y_pred, probabilities


def _val_record(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_f1": {
            name: values["f1"] for name, values in metrics["per_class"].items()
        },
    }


def _write_test_predictions(
    logger: RunLogger,
    dataset: WasteDataset,
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: Sequence[str],
) -> Path:
    """Ghi predictions cho các mẫu đã đánh giá (hỗ trợ cả limit-batches)."""
    samples = dataset.samples[: len(y_true)]
    return write_predictions(
        logger.predictions / "test.csv",
        paths=[sample.path for sample in samples],
        datasets=[sample.dataset for sample in samples],
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
    )


def _run(
    logger: RunLogger,
    config: dict[str, Any],
    seed: int,
    limit_batches: int | None,
    resume: bool,
) -> TrainingResult:
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    train_cfg = config["training"]

    class_names = tuple(data_cfg.get("class_names") or CANONICAL_CLASSES)
    data_root = _resolve_data_root(config)
    split_csv = _resolve_split_csv(config)
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    augmentation = bool(data_cfg.get("augmentation", True))
    num_classes = int(model_cfg.pop("num_classes", len(class_names)))
    max_steps = limit_batches if limit_batches is not None else train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps else None
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience_limit = int(train_cfg.get("early_stopping_patience", 7))
    monitor = str(train_cfg.get("monitor", "val_macro_f1"))
    if monitor != "val_macro_f1":
        raise ValueError("Hiện chỉ hỗ trợ monitor='val_macro_f1'")
    grad_clip = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    evaluate_test = bool(train_cfg.get("evaluate_test", True))
    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_dataset = WasteDataset(
        data_root=data_root,
        split_csv=split_csv,
        split="train",
        class_names=class_names,
        transform=build_transforms(
            train=True, image_size=image_size, augmentation=augmentation
        ),
    )
    val_dataset = WasteDataset(
        data_root=data_root,
        split_csv=split_csv,
        split="val",
        class_names=class_names,
        transform=build_transforms(train=False, image_size=image_size),
    )
    test_dataset = (
        WasteDataset(
            data_root=data_root,
            split_csv=split_csv,
            split="test",
            class_names=class_names,
            transform=build_transforms(train=False, image_size=image_size),
        )
        if evaluate_test
        else None
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )
    test_loader = (
        _make_loader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        if test_dataset is not None
        else None
    )

    model_name = str(model_cfg.pop("name"))
    model = build_model(model_name, num_classes=num_classes, **model_cfg)
    freeze_epochs = int(model_cfg.get("freeze_epochs", 0) or 0)
    unfreeze_blocks = int(model_cfg.get("unfreeze_blocks", 2) or 2)
    if freeze_epochs > 0 and hasattr(model, "features"):
        freeze_backbone(model)

    model.to(device)
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.update(
        model={
            "name": model_name,
            "pretrained": bool(model_cfg.get("pretrained", False)),
            "params": total_params,
            "trainable_params": trainable_params,
            "freeze_epochs": freeze_epochs,
            "unfreeze_blocks": unfreeze_blocks,
        },
        dataset={
            "splits_version": data_cfg.get("splits_version", "v1"),
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset) if test_dataset is not None else None,
            "data_root": str(data_root),
            "split_csv": str(split_csv),
        },
        device=str(device),
    )

    if data_cfg.get("class_weighting"):
        weights = _class_weights(train_dataset, clip=(0.5, 3.0)).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, epochs, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_metric = float("-inf")
    best_epoch: int | None = None
    best_val: dict[str, Any] = {}
    if resume:
        if freeze_epochs > 0:
            raise NotImplementedError("Resume với freeze_epochs > 0 chưa được hỗ trợ.")
        last_path = logger.checkpoints / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"Không có checkpoint để resume: {last_path}")
        meta = load_last(
            last_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(meta.epoch or 0) + 1
        best_metric = float(meta.best_metric or float("-inf"))
        best_epoch = meta.epoch

    git_commit = env_utils.git_info()["commit"]
    logger.log_event(
        "training_started",
        epochs=epochs,
        device=str(device),
        params=total_params,
        batch_size=batch_size,
        max_steps=max_steps,
    )

    history: list[dict[str, Any]] = []
    patience = 0
    unfrozen = freeze_epochs == 0
    train_started = time.monotonic()

    for epoch in range(start_epoch, epochs + 1):
        if (
            not unfrozen
            and epoch == freeze_epochs + 1
            and hasattr(model, "features")
        ):
            unfreeze_last_blocks(model, unfreeze_blocks)
            optimizer = _make_optimizer(model, lr, weight_decay)
            scheduler = _make_scheduler(optimizer, epochs - epoch + 1, lr)
            unfrozen = True
            logger.log_event(
                "unfroze_backbone",
                epoch=epoch,
                blocks=unfreeze_blocks,
                trainable_params=count_parameters(model, trainable_only=True),
            )

        epoch_started = time.monotonic()
        train_metrics = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            class_names,
            use_amp=use_amp,
            max_steps=max_steps,
            grad_clip=grad_clip,
        )
        train_duration = time.monotonic() - epoch_started

        val_started = time.monotonic()
        val_metrics, _, _, _ = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            class_names,
            use_amp=use_amp,
            max_steps=max_steps,
        )
        val_duration = time.monotonic() - val_started
        val_record = _val_record(val_metrics)

        logger.log_epoch(
            epoch,
            "train",
            {
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "macro_f1": train_metrics["macro_f1"],
            },
            duration_sec=train_duration,
        )
        logger.log_epoch(epoch, "val", val_record, duration_sec=val_duration)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_record})

        last_path = logger.checkpoints / "last.pt"
        save_last(
            last_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=epoch * len(train_loader),
            best_metric=best_metric if best_metric > float("-inf") else None,
            class_names=list(class_names),
            config=config,
            seed=seed,
            scheduler=scheduler,
            scaler=scaler,
            git_commit=git_commit,
        )
        logger.log_checkpoint("last", last_path, epoch=epoch, metric=val_record["macro_f1"])

        if val_metrics["macro_f1"] > best_metric:
            best_metric = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_val = {"epoch": epoch, **val_record}
            best_path = logger.checkpoints / "best.pt"
            save_best(
                best_path,
                model,
                epoch=epoch,
                metric=best_metric,
                class_names=list(class_names),
                config=config,
                seed=seed,
                git_commit=git_commit,
            )
            logger.log_checkpoint("best", best_path, epoch=epoch, metric=best_metric)
            patience = 0
        else:
            patience += 1

        scheduler.step()
        if patience >= patience_limit:
            logger.log_event("early_stopped", epoch=epoch, best_epoch=best_epoch)
            break

    train_time = time.monotonic() - train_started
    best_path = logger.checkpoints / "best.pt"
    if not best_path.is_file():
        raise RuntimeError("Không có best checkpoint sau khi train.")

    test_metrics: dict[str, Any] | None = None
    inference_time = 0.0
    if test_dataset is not None and test_loader is not None:
        load_best(best_path, model, map_location=device)
        inference_started = time.monotonic()
        test_metrics, y_true, y_pred, probabilities = _evaluate(
            model,
            test_loader,
            criterion,
            device,
            class_names,
            use_amp=use_amp,
            max_steps=max_steps,
            collect_probabilities=True,
        )
        inference_time = time.monotonic() - inference_started
        _write_test_predictions(
            logger,
            test_dataset,
            y_true,
            y_pred,
            probabilities,
            class_names,
        )

    efficiency = efficiency_metrics(
        params=total_params,
        train_time_sec=train_time,
        inference_time_sec=inference_time,
        n_images=len(test_dataset) if test_dataset is not None else 0,
    )
    logger.finish(
        {
            "class_names": list(class_names),
            "best_val": best_val,
            "test": test_metrics,
            "efficiency": efficiency,
        }
    )
    logger.log_event("training_finished", best_epoch=best_epoch, best_val_macro_f1=best_metric)
    return TrainingResult(
        run_dir=logger.root,
        best_val=best_val,
        test=test_metrics,
        history=history,
    )


def run_training(
    *,
    config_path: Path,
    plan_path: Path | None,
    seed: int | None = None,
    run_id: str | None = None,
    epochs: int | None = None,
    limit_batches: int | None = None,
    resume: bool = False,
    allow_unplanned: bool = False,
    experiments_root: Path | None = None,
) -> TrainingResult:
    """Chạy một thí nghiệm hoàn chỉnh từ config + plan."""
    config = load_config(config_path)
    resolved_seed = int(seed if seed is not None else config["seed"])
    config["seed"] = resolved_seed
    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    experiment = str(config["experiment"])
    resolved_run_id = run_id or f"{experiment}_s{resolved_seed}"

    logger, _ = start_run(
        config=config,
        run_id=resolved_run_id,
        seed=resolved_seed,
        plan_path=plan_path,
        rq=str(config["rq"]),
        experiments_root=experiments_root,
        resume=resume,
        allow_unplanned=allow_unplanned,
    )
    _set_seed(resolved_seed)

    with logger:
        return _run(logger, config, resolved_seed, limit_batches, resume)
