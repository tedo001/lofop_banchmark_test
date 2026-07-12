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
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lofop import Detector
from lofop.data import load_dataset
from lofop.data.synthetic import generate_shapes_dataset

from lofop_bench.progress import TrainingProgress


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


def run_accuracy(
    output_dir: str | Path,
    *,
    variant: str = "n",
    device: str = "auto",
    epochs: int = 20,
    image_size: int = 96,
    batch_size: int = 8,
    lr: float = 0.01,
    seed: int = 0,
    # Real-dataset mode (all four together); omit to use synthetic shapes.
    data_format: str | None = None,
    train_source: str | None = None,
    val_source: str | None = None,
    image_root: str | None = None,
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

    if data_format and train_source:
        dataset_name = data_format
        kwargs = {"image_root": image_root} if image_root else {}
        val = load_dataset(data_format, val_source or train_source, **kwargs)
        num_classes = len(val.categories)
        detector = Detector(variant, num_classes=num_classes, image_size=image_size, device=device)
        with TrainingProgress(epochs, prefix=f"  training {variant} on {dataset_name}"):
            detector.train(
                data_format=data_format, train_source=train_source, val_source=val_source,
                image_root=image_root, epochs=epochs, batch_size=batch_size, lr=lr,
                workers=0, checkpoint_dir=output_dir / "checkpoints",
            )
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
        with TrainingProgress(epochs, prefix=f"  training {variant} on {dataset_name}"):
            detector.train(
                train_data=train, val_data=val, epochs=epochs, batch_size=batch_size, lr=lr,
                workers=0, checkpoint_dir=output_dir / "checkpoints",
            )

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
    return result
