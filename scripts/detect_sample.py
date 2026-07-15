#!/usr/bin/env python3
"""Run a trained LOFOP detector on sample images and save annotated pictures.

This is the "see it detect" step: after training on COCO128 (which leaves a
checkpoint at ``results/checkpoints/best.pt``), run the detector on a few
validation images and write pictures with the predicted boxes drawn on them to
``results/detections/``.

    python scripts/detect_sample.py --data-config configs/coco128.yaml
    python scripts/detect_sample.py --images path/to/a.jpg path/to/b.jpg --num-classes 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from lofop import Detector
from lofop.data import draw_boxes, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "results" / "checkpoints" / "best.pt"


def _val_images(data_config: Path, limit: int) -> tuple[list[Path], list[str], int]:
    """Sample image paths, class names, and class count from a dataset config."""
    cfg = yaml.safe_load(Path(data_config).read_text(encoding="utf-8"))
    kwargs = {"image_root": cfg["image_root"]} if cfg.get("image_root") else {}
    dataset = load_dataset(cfg["data_format"], cfg.get("val_source") or cfg["train_source"], **kwargs)
    names = [category.name for category in dataset.categories]
    images = [dataset.image_path(sample) for sample in dataset.samples[:limit]]
    return images, names, len(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect on sample images and draw boxes.")
    parser.add_argument("--data-config", type=Path, default=None,
                        help="dataset YAML; samples its val images")
    parser.add_argument("--images", nargs="*", type=Path, default=None,
                        help="explicit image paths (instead of --data-config)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--variant", default="n")
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=None,
                        help="lower (e.g. 0.4) merges overlapping duplicate boxes harder")
    parser.add_argument("--limit", type=int, default=6, help="images to sample from --data-config")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "detections")
    args = parser.parse_args(argv)

    class_names = [f"class_{i}" for i in range(args.num_classes)]
    if args.images:
        images = list(args.images)
    elif args.data_config:
        images, class_names, args.num_classes = _val_images(args.data_config, args.limit)
    else:
        print("Provide --data-config or --images", file=sys.stderr)
        return 2

    if not args.checkpoint.is_file():
        print(
            f"No checkpoint at {args.checkpoint}. Train first, e.g.\n"
            f"  python run_benchmarks.py --data-config configs/coco128.yaml "
            f"--variant {args.variant} --epochs 100 --acc-size {args.size} --skip-latency",
            file=sys.stderr,
        )
        return 1

    detector = Detector(
        args.variant, num_classes=args.num_classes, checkpoint=args.checkpoint,
        class_names=class_names, image_size=args.size,
    )
    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    if args.nms_iou is not None:
        detector.model.nms_iou = args.nms_iou
    detections = detector.predict(images, score_threshold=args.score_threshold)
    for image_path, result in zip(images, detections):
        with Image.open(image_path) as image:
            annotated = draw_boxes(
                image, result.boxes, labels=result.labels, scores=result.scores,
                class_names=detector.class_names,
            )
        out_path = args.out / f"{Path(image_path).stem}_pred.png"
        annotated.save(out_path)
        print(f"  {out_path}  ({len(result.boxes)} detections)", file=sys.stderr)

    print(f"Annotated images written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
