"""Accuracy benchmark: train-and-evaluate, on synthetic data or a real dataset.

Two modes:

* **Synthetic (default):** a short, fully deterministic run on LOFOP's built-in
  ``shapes`` dataset -- a reproducible regression instrument, not a real-world
  claim.
* **Real dataset:** point it at a COCO-format dataset and it trains a variant
  and reports genuine mAP/precision/recall/F1. On a CUDA GPU (e.g. an RTX 4060)
  training uses AMP automatically and is dramatically faster.

All randomness is seeded so the synthetic mode is stable across runs.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lofop import Detector
from lofop.data import load_dataset
from lofop.data.synthetic import generate_shapes_dataset
from lofop.registries import EVENTS

from lofop_bench.progress import TrainingProgress


class _MetricHistory:
    """Collect per-epoch loss and mAP by listening to ``train.epoch_end``.

    Lets a run produce an mAP-vs-epoch curve, which is the honest diagnostic for
    "should I train longer?": still climbing means keep going; flat means this
    model has plateaued.
    """

    def __init__(self) -> None:
        self.epochs: list[dict] = []
        self._subscription = None

    def __enter__(self) -> _MetricHistory:
        self._subscription = EVENTS.subscribe("train.epoch_end", self._record)
        return self

    def _record(self, event) -> None:
        metrics = event.get("metrics") or {}
        self.epochs.append({
            "epoch": int(event.get("epoch", 0)) + 1,
            "loss": event.get("loss"),
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
        })

    def __exit__(self, *exc_info) -> None:
        if self._subscription is not None:
            EVENTS.unsubscribe(self._subscription)


@dataclass(frozen=True)
class AccuracyResult:
    """One accuracy benchmark run and the settings that produced it."""

    variant: str
    dataset: str
    device: str
    epochs: int
    image_size: int
    num_classes: int
    seed: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float

    def to_markdown(self) -> str:
        return (
            "# LOFOP accuracy benchmark\n\n"
            f"Variant `lofop-detect-{self.variant}` on `{self.dataset}`, "
            f"{self.epochs} epochs at {self.image_size}px on `{self.device}` "
            f"({self.num_classes} classes, seed {self.seed}).\n\n"
            "| Metric | Value |\n|---|---:|\n"
            f"| mAP@50 | {self.map50:.4f} |\n"
            f"| mAP@50:95 | {self.map50_95:.4f} |\n"
            f"| Precision | {self.precision:.4f} |\n"
            f"| Recall | {self.recall:.4f} |\n"
            f"| F1 | {self.f1:.4f} |\n"
        )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _subset(dataset, limit: int | None):
    """Take the first ``limit`` samples for a faster run (categories unchanged)."""
    if not limit or limit >= len(dataset):
        return dataset
    from lofop.data.dataset import Dataset

    return Dataset(
        dataset.name, dataset.categories, dataset.samples[:limit], image_root=dataset.image_root
    )


def _steps_per_epoch(num_images: int, batch_size: int) -> int:
    """Batches per epoch, mirroring the trainer's drop-last DataLoader."""
    return max(num_images // batch_size, 1) if num_images > batch_size else 1


def _train_with_progress(detector, progress, train_kwargs) -> None:
    """Run ``detector.train`` with per-batch loss feeding the training log.

    Wraps the model's ``compute_losses`` (called exactly once per training
    batch) so iteration lines show the real batch loss. Per-epoch evaluation
    runs on the EMA deepcopy, so eval passes are not counted. The wrapper only
    converts the loss tensor on the iterations that print, so there is no
    per-batch GPU sync overhead between logs.
    """
    model = detector.model
    original = model.compute_losses

    def instrumented(images, targets):
        losses = original(images, targets)
        progress.on_batch(losses["total"].detach())
        return losses

    model.compute_losses = instrumented
    try:
        with progress:
            detector.train(**train_kwargs)
    finally:
        del model.compute_losses  # drop the instance shadow; class method returns


def run_accuracy(
    output_dir: str | Path,
    *,
    variant: str = "n",
    device: str = "auto",
    epochs: int = 20,
    image_size: int = 96,
    batch_size: int = 8,
    lr: float = 0.01,
    workers: int = 0,
    seed: int = 0,
    # Real-dataset mode (all four together); omit to use synthetic shapes.
    data_format: str | None = None,
    train_source: str | None = None,
    val_source: str | None = None,
    image_root: str | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
    # Synthetic-mode sizing.
    train_images: int = 48,
    val_images: int = 16,
    basename: str = "accuracy",
) -> AccuracyResult:
    """Train a variant, evaluate it, and write ``accuracy.{md,json}``."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)
    output_dir = Path(output_dir)

    history = _MetricHistory()
    if data_format and train_source:
        dataset_name = data_format
        kwargs = {"image_root": image_root} if image_root else {}
        train = _subset(load_dataset(data_format, train_source, **kwargs), limit_train)
        val = _subset(load_dataset(data_format, val_source or train_source, **kwargs), limit_val)
        num_classes = len(val.categories)
        print(f"  dataset: {len(train)} train / {len(val)} val images, "
              f"{num_classes} classes", file=sys.stderr)
        detector = Detector(variant, num_classes=num_classes, image_size=image_size, device=device)
        progress = TrainingProgress(
            epochs, prefix=f"  training {variant} on {dataset_name}",
            steps_per_epoch=_steps_per_epoch(len(train), batch_size),
        )
        with history:
            _train_with_progress(detector, progress, dict(
                train_data=train, val_data=val, epochs=epochs, batch_size=batch_size, lr=lr,
                workers=workers, checkpoint_dir=output_dir / "checkpoints",
            ))
    else:
        dataset_name = "shapes (synthetic)"
        workdir = output_dir / "_shapes_data"
        train = generate_shapes_dataset(
            workdir / "train", num_images=train_images, image_size=image_size, seed=seed
        )
        val = generate_shapes_dataset(
            workdir / "val", num_images=val_images, image_size=image_size, seed=seed + 1
        )
        num_classes = len(train.categories)
        detector = Detector(variant, num_classes=num_classes, image_size=image_size, device=device)
        progress = TrainingProgress(
            epochs, prefix=f"  training {variant} on {dataset_name}",
            steps_per_epoch=_steps_per_epoch(len(train), batch_size),
        )
        with history:
            _train_with_progress(detector, progress, dict(
                train_data=train, val_data=val, epochs=epochs, batch_size=batch_size, lr=lr,
                workers=workers, checkpoint_dir=output_dir / "checkpoints",
            ))

    metrics = detector.evaluate(val)
    result = AccuracyResult(
        variant=variant, dataset=dataset_name, device=device, epochs=epochs,
        image_size=image_size, num_classes=num_classes, seed=seed,
        map50=metrics.map50, map50_95=metrics.map50_95,
        precision=metrics.precision, recall=metrics.recall, f1=metrics.f1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.md").write_text(result.to_markdown(), encoding="utf-8")
    (output_dir / f"{basename}.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{basename}_history.json").write_text(
        json.dumps(history.epochs, indent=2) + "\n", encoding="utf-8"
    )
    return result
