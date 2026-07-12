#!/usr/bin/env python3
"""Download real COCO images and prepare them for LOFOP training.

Fetches the COCO val2017 set (5,000 real, well-annotated images -- proper
"quality data") and its instance annotations, optionally keeps only the classes
you ask for, splits into train/val, and writes COCO-format JSON plus
``configs/coco.yaml``::

    data/coco/images/*.jpg
    data/coco/train.json
    data/coco/val.json

Then train on it exactly like the small set::

    python run_benchmarks.py --device cuda --data-config configs/coco.yaml \
        --variant s --epochs 100 --acc-size 640 --skip-latency --skip-structural

This is ~1 GB of images plus ~250 MB of annotations, so the first run downloads
for a few minutes. Use ``--classes person`` and/or ``--max-images 1500`` for a
faster first real result; drop them for the full set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGES_URL = os.environ.get(
    "LOFOP_COCO_IMAGES_URL", "http://images.cocodataset.org/zips/val2017.zip"
)
ANN_URL = os.environ.get(
    "LOFOP_COCO_ANN_URL",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
)


def _download_to(url: str, path: Path) -> None:
    """Stream a (possibly large) download to disk with a byte counter."""
    if path.is_file() and path.stat().st_size > 0:
        print(f"Using cached {path.name}", file=sys.stderr)
        return
    print(f"Downloading {url} ...", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (lofop-bench)"})
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = f"{done / total * 100:5.1f}%" if total else f"{done / 1e6:.0f} MB"
            print(f"\r  {pct} ({done / 1e6:.0f} MB)", end="", file=sys.stderr)
    print(file=sys.stderr)


def _filter_and_split(
    coco: dict, class_names: list[str] | None, val_fraction: float,
    max_images: int | None, seed: int,
) -> tuple[dict, dict]:
    """Optionally keep only some classes, then split images into train/val."""
    categories = coco["categories"]
    if class_names:
        wanted = {c["id"] for c in categories if c["name"] in class_names}
        remap = {old: i + 1 for i, old in enumerate(sorted(wanted))}
        categories = [
            {"id": remap[c["id"]], "name": c["name"]} for c in categories if c["id"] in wanted
        ]
        annotations = [
            {**a, "category_id": remap[a["category_id"]]}
            for a in coco["annotations"] if a["category_id"] in wanted
        ]
    else:
        annotations = coco["annotations"]

    kept_image_ids = {a["image_id"] for a in annotations}
    images = [im for im in coco["images"] if im["id"] in kept_image_ids]
    random.Random(seed).shuffle(images)
    if max_images:
        images = images[:max_images]

    keep_ids = {im["id"] for im in images}
    annotations = [a for a in annotations if a["image_id"] in keep_ids]
    split = max(1, int(len(images) * (1 - val_fraction)))
    train_images, val_images = images[:split], images[split:] or images[-1:]

    def _subset(subset_images: list[dict]) -> dict:
        ids = {im["id"] for im in subset_images}
        return {
            "images": subset_images,
            "annotations": [a for a in annotations if a["image_id"] in ids],
            "categories": categories,
        }

    return _subset(train_images), _subset(val_images)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real COCO data for LOFOP training.")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "coco")
    parser.add_argument("--classes", nargs="+", default=None, help="keep only these class names")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None, help="cap for a faster first run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    images_zip = args.dest / "val2017.zip"
    ann_zip = args.dest / "annotations.zip"
    _download_to(IMAGES_URL, images_zip)
    _download_to(ANN_URL, ann_zip)

    image_dir = args.dest / "images"
    if not image_dir.is_dir():
        print("Extracting images ...", file=sys.stderr)
        with zipfile.ZipFile(images_zip) as archive:
            archive.extractall(args.dest)
        extracted = next(args.dest.glob("val2017"), None) or args.dest
        extracted.rename(image_dir)
    ann_json = args.dest / "instances_val2017.json"
    if not ann_json.is_file():
        print("Extracting annotations ...", file=sys.stderr)
        with zipfile.ZipFile(ann_zip) as archive:
            member = next(n for n in archive.namelist() if n.endswith("instances_val2017.json"))
            with archive.open(member) as src, open(ann_json, "wb") as dst:
                shutil.copyfileobj(src, dst)

    coco = json.loads(ann_json.read_text(encoding="utf-8"))
    if args.classes:
        names = {c["name"] for c in coco["categories"]}
        unknown = [c for c in args.classes if c not in names]
        if unknown:
            print(f"Unknown class name(s): {unknown}", file=sys.stderr)
            return 2
    train, val = _filter_and_split(
        coco, args.classes, args.val_fraction, args.max_images, args.seed
    )
    (args.dest / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (args.dest / "val.json").write_text(json.dumps(val), encoding="utf-8")

    config = REPO_ROOT / "configs" / "coco.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    scope = f"classes {args.classes}" if args.classes else "all 80 classes"
    config.write_text(
        f"# Auto-generated by scripts/get_coco.py. COCO val2017 ({scope}).\n"
        "data_format: coco\n"
        f"train_source: {(args.dest / 'train.json').relative_to(REPO_ROOT)}\n"
        f"val_source: {(args.dest / 'val.json').relative_to(REPO_ROOT)}\n"
        f"image_root: {image_dir.relative_to(REPO_ROOT)}\n",
        encoding="utf-8",
    )
    print(
        f"COCO ready ({scope}): {len(train['images'])} train / {len(val['images'])} val "
        f"images, {len(train['categories'])} classes at {args.dest}\n"
        f"Config written to {config}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
