#!/usr/bin/env python3
"""Download COCO128 and write it as a LOFOP-ready COCO-format dataset.

COCO128 is a 128-image subset of COCO (~6 MB) -- the standard "does my pipeline
work at all" dataset. This script downloads it, optionally keeps only the
classes you ask for, splits it into train/val, and writes COCO-format
annotation JSON that LOFOP loads directly::

    data/coco128/images/*.jpg
    data/coco128/train.json
    data/coco128/val.json

plus ``configs/coco128.yaml`` pointing at them. Run once, then benchmark with
``--data-config configs/coco128.yaml``.

Tip: 80 classes over ~100 images is too little to learn from scratch. Narrow
the task to make the small set tractable, e.g. ``--classes person``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

# Public download location of the 128-image COCO subset (~6 MB). Override with
# the LOFOP_COCO128_URL environment variable to host it yourself.
COCO128_URL = os.environ.get(
    "LOFOP_COCO128_URL",
    "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
)

# The 80 COCO class names, in canonical index order.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

REPO_ROOT = Path(__file__).resolve().parent.parent


def _download(url: str) -> bytes:
    print(f"Downloading {url} ...", file=sys.stderr)
    # A browser-like User-Agent avoids a 403 for the default urllib agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (lofop-bench)"})
    with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted release asset)
        return response.read()


def _find_split_dirs(extracted: Path) -> tuple[Path, Path]:
    """Locate the image and label directories inside the extracted archive."""
    images = next(extracted.rglob("images"))
    image_dir = next((p for p in images.rglob("*") if p.is_dir()), images)
    label_dir = Path(str(image_dir).replace("images", "labels", 1))
    return image_dir, label_dir


def _boxes(label_file: Path, remap: dict[int, int] | None) -> list[tuple[int, float, float, float, float]]:
    """Parse a normalized ``cls cx cy w h`` label file into kept, remapped rows."""
    if not label_file.is_file():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        if remap is None:
            rows.append((cls, cx, cy, bw, bh))
        elif cls in remap:
            rows.append((remap[cls], cx, cy, bw, bh))
    return rows


def _build_split(
    stems: list[str], image_dir: Path, label_dir: Path,
    class_names: list[str], remap: dict[int, int] | None, images_out: Path,
) -> dict:
    """Copy images and build a COCO-format dict for one split."""
    images_out.mkdir(parents=True, exist_ok=True)
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for image_id, stem in enumerate(stems):
        boxes = _boxes(label_dir / f"{stem}.txt", remap)
        if remap is not None and not boxes:
            continue  # drop images that contain none of the requested classes
        image_path = next(image_dir.glob(f"{stem}.*"), None)
        if image_path is None:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        shutil.copy(image_path, images_out / image_path.name)
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        for cls, cx, cy, bw, bh in boxes:
            x, y = (cx - bw / 2) * width, (cy - bh / 2) * height
            w, h = bw * width, bh * height
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": cls + 1,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0,
            })
            ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch COCO128 for LOFOP benchmarking.")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "coco128")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="portion held out for val")
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="keep only these class names (e.g. --classes person). Narrows the "
             "task so training on the tiny set actually learns.",
    )
    args = parser.parse_args(argv)

    if args.classes:
        unknown = [c for c in args.classes if c not in COCO_CLASSES]
        if unknown:
            print(f"Unknown class name(s): {unknown}", file=sys.stderr)
            return 2
        remap = {COCO_CLASSES.index(name): i for i, name in enumerate(args.classes)}
        class_names = list(args.classes)
    else:
        remap = None
        class_names = COCO_CLASSES

    raw = REPO_ROOT / "data" / "_coco128_raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(_download(COCO128_URL))) as archive:
        archive.extractall(raw)

    image_dir, label_dir = _find_split_dirs(raw)
    stems = sorted(p.stem for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    if remap is not None:
        stems = [s for s in stems if _boxes(label_dir / f"{s}.txt", remap)]
    if not stems:
        print("No matching images found in the archive", file=sys.stderr)
        return 1
    split = max(1, int(len(stems) * (1 - args.val_fraction)))
    train_stems, val_stems = stems[:split], stems[split:] or stems[-1:]

    if args.dest.exists():
        shutil.rmtree(args.dest)
    images_out = args.dest / "images"
    train = _build_split(train_stems, image_dir, label_dir, class_names, remap, images_out)
    val = _build_split(val_stems, image_dir, label_dir, class_names, remap, images_out)
    (args.dest / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (args.dest / "val.json").write_text(json.dumps(val), encoding="utf-8")

    config = REPO_ROOT / "configs" / "coco128.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    scope = f"classes {class_names}" if remap is not None else "all 80 classes"
    config.write_text(
        f"# Auto-generated by scripts/get_coco128.py. COCO128 subset ({scope}).\n"
        "data_format: coco\n"
        f"train_source: {(args.dest / 'train.json').relative_to(REPO_ROOT)}\n"
        f"val_source: {(args.dest / 'val.json').relative_to(REPO_ROOT)}\n"
        f"image_root: {images_out.relative_to(REPO_ROOT)}\n",
        encoding="utf-8",
    )
    shutil.rmtree(raw, ignore_errors=True)
    print(
        f"COCO128 ready ({scope}): {len(train['images'])} train / "
        f"{len(val['images'])} val images at {args.dest}\nConfig written to {config}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
