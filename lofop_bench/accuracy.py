"""Accuracy benchmark: a short, deterministic train-and-evaluate run.

Trains a LOFOP-Detect variant on the framework's built-in synthetic ``shapes``
dataset for a handful of CPU epochs and reports the evaluation metrics. This is
a controlled instrument, not a real-world accuracy claim -- its value is being
*reproducible*, so it catches quality regressions between LOFOP versions. Every
source of randomness is seeded (torch and the stdlib ``random`` used by the
augmentation flip), so a given (seed, epochs, size) yields stable numbers.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from lofop import Detector
from lofop.data.synthetic import generate_shapes_dataset


@dataclass(frozen=True)
class AccuracyResult:
    """One accuracy benchmark run and the settings that produced it."""

    variant: str
    epochs: int
    image_size: int
    train_images: int
    val_images: int
    seed: int
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1: float

    def to_markdown(self) -> str:
        return (
            f"# LOFOP accuracy benchmark\n\n"
            f"Variant `lofop-detect-{self.variant}`, {self.epochs} CPU epochs at "
            f"{self.image_size}px on the synthetic `shapes` dataset "
            f"({self.train_images} train / {self.val_images} val images, seed {self.seed}).\n\n"
            f"| Metric | Value |\n|---|---:|\n"
            f"| mAP@50 | {self.map50:.4f} |\n"
            f"| mAP@50:95 | {self.map50_95:.4f} |\n"
            f"| Precision | {self.precision:.4f} |\n"
            f"| Recall | {self.recall:.4f} |\n"
            f"| F1 | {self.f1:.4f} |\n\n"
            f"Reproducible instrument on synthetic data, not a real-world accuracy claim.\n"
        )


def run_accuracy(
    output_dir: str | Path,
    *,
    variant: str = "n",
    epochs: int = 20,
    image_size: int = 96,
    train_images: int = 48,
    val_images: int = 16,
    seed: int = 0,
    basename: str = "accuracy",
) -> AccuracyResult:
    """Train on synthetic shapes, evaluate, and write ``accuracy.{md,json}``."""
    random.seed(seed)
    torch.manual_seed(seed)

    workdir = Path(output_dir) / "_shapes_data"
    train = generate_shapes_dataset(
        workdir / "train", num_images=train_images, image_size=image_size, seed=seed
    )
    val = generate_shapes_dataset(
        workdir / "val", num_images=val_images, image_size=image_size, seed=seed + 1
    )

    detector = Detector(
        variant, num_classes=len(train.categories), image_size=image_size, device="cpu"
    )
    detector.train(
        train_data=train, val_data=val, epochs=epochs, batch_size=8, lr=0.01,
        workers=0, checkpoint_dir=Path(output_dir) / "_run",
    )
    metrics = detector.evaluate(val)

    result = AccuracyResult(
        variant=variant, epochs=epochs, image_size=image_size,
        train_images=train_images, val_images=val_images, seed=seed,
        map50=metrics.map50, map50_95=metrics.map50_95,
        precision=metrics.precision, recall=metrics.recall, f1=metrics.f1,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.md").write_text(result.to_markdown(), encoding="utf-8")
    (output_dir / f"{basename}.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(run_accuracy(Path(__file__).resolve().parent.parent / "results"))
