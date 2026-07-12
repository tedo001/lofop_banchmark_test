#!/usr/bin/env python3
"""Download real COCO images and prepare them for LOFOP training.

Fetches COCO 2017 images and their instance annotations, optionally keeps only
the classes you ask for, splits into train/val, and writes COCO-format JSON plus
``configs/coco.yaml``::

    data/coco/images/*.jpg
    data/coco/train.json
    data/coco/val.json

Two ways to size the download:

* ``--split val2017`` (default) grabs the 5,000-image val set as one ~1 GB zip.
* ``--split train2017 --max-gb 5`` streams images one at a time from the big
  training set and stops at the size budget, so you get a ~4-5 GB slice without
  the full ~18 GB. Combine with ``--classes`` to download only what you need.

Then train exactly like the small set::

    python run_benchmarks.py --device cuda --data-config configs/coco.yaml \
        --variant s --epochs 100 --acc-size 640 --skip-latency --skip-structural
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VAL_ZIP_URL = os.environ.get(
    "LOFOP_COCO_IMAGES_URL", "http://images.cocodataset.org/zips/val2017.zip"
)
ANN_URL = os.environ.get(
    "LOFOP_COCO_ANN_URL",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
)
IMAGE_BASE = "http://images.cocodataset.org"  # /<split>/<file_name>
_UA = {"User-Agent": "Mozilla/5.0 (lofop-bench)"}


def _download_to(url: str, path: Path) -> None:
    """Stream a (possibly large) download to disk with a byte counter."""
    if path.is_file() and path.stat().st_size > 0:
        print(f"Using cached {path.name}", file=sys.stderr)
        return
    print(f"Downloading {url} ...", file=sys.stderr)
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request) as response, open(path, "wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = f"{done / total * 100:5.1f}%" if total else f"{done / 1e6:.0f} MB"
            print(f"\r  {pct} ({done / 1e6:.0f} MB)", end="", file=sys.stderr)
    print(file=sys.stderr)


def _fetch_image(split: str, file_name: str, image_dir: Path) -> int:
    dst = image_dir / file_name
    if dst.is_file() and dst.stat().st_size > 0:
        return dst.stat().st_size
    request = urllib.request.Request(f"{IMAGE_BASE}/{split}/{file_name}", headers=_UA)
    with urllib.request.urlopen(request) as response:  # noqa: S310 (public dataset host)
        data = response.read()
    dst.write_bytes(data)
    return len(data)


def _download_images_individually(
    file_names: list[str], split: str, image_dir: Path,
    max_images: int | None, max_gb: float | None, workers: int = 16,
) -> set[str]:
    """Download images concurrently until a count/size budget is hit.

    ``max_images`` is a hard cap on how many downloads are even started, so it
    is exact. A byte budget (``max_gb``) may overshoot by the in-flight batch
    (a few tens of images), which is negligible at multi-GB scale.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    budget = int(max_gb * 1e9) if max_gb else None
    names = iter(file_names)
    submitted = 0
    got: set[str] = set()
    total = 0

    def take(count: int) -> list[str]:
        nonlocal submitted
        batch: list[str] = []
        while len(batch) < count and not (max_images and submitted >= max_images):
            try:
                batch.append(next(names))
            except StopIteration:
                break
            submitted += 1
        return batch

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(_fetch_image, split, name, image_dir): name
            for name in take(workers * 2)
        }
        while pending:
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    total += future.result()
                    got.add(name)
                except Exception:  # noqa: BLE001 (skip a bad image, keep going)
                    continue
            print(f"\r  {len(got)} images, {total / 1e9:.2f} GB", end="", file=sys.stderr)
            if budget and total >= budget:
                for future in pending:
                    future.cancel()
                break
            for name in take(len(done)):
                pending[pool.submit(_fetch_image, split, name, image_dir)] = name
    print(file=sys.stderr)
    return got


def _filter(coco: dict, class_names: list[str] | None) -> tuple[list, list, list]:
    """Return (categories, annotations, candidate_images) after optional class filter."""
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
    kept_ids = {a["image_id"] for a in annotations}
    images = [im for im in coco["images"] if im["id"] in kept_ids]
    return categories, annotations, images


def _split(images: list, annotations: list, categories: list, val_fraction: float) -> tuple:
    split = max(1, int(len(images) * (1 - val_fraction)))
    train_images, val_images = images[:split], images[split:] or images[-1:]

    def subset(subset_images: list) -> dict:
        ids = {im["id"] for im in subset_images}
        return {
            "images": subset_images,
            "annotations": [a for a in annotations if a["image_id"] in ids],
            "categories": categories,
        }

    return subset(train_images), subset(val_images)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch real COCO data for LOFOP training.")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "coco")
    parser.add_argument("--split", choices=["val2017", "train2017"], default="val2017")
    parser.add_argument("--classes", nargs="+", default=None, help="keep only these class names")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-images", type=int, default=None, help="cap the image count")
    parser.add_argument("--max-gb", type=float, default=None,
                        help="stop downloading images at about this many GB (train2017)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    image_dir = args.dest / "images"

    # Annotations for the chosen split.
    ann_zip = args.dest / "annotations.zip"
    _download_to(ANN_URL, ann_zip)
    ann_json = args.dest / f"instances_{args.split}.json"
    if not ann_json.is_file():
        with zipfile.ZipFile(ann_zip) as archive:
            member = next(n for n in archive.namelist() if n.endswith(f"instances_{args.split}.json"))
            with archive.open(member) as src, open(ann_json, "wb") as dst:
                shutil.copyfileobj(src, dst)

    coco = json.loads(ann_json.read_text(encoding="utf-8"))
    if args.classes:
        names = {c["name"] for c in coco["categories"]}
        unknown = [c for c in args.classes if c not in names]
        if unknown:
            print(f"Unknown class name(s): {unknown}", file=sys.stderr)
            return 2

    categories, annotations, images = _filter(coco, args.classes)
    random.Random(args.seed).shuffle(images)

    # Acquire the actual image files.
    use_zip = args.split == "val2017" and args.max_gb is None and args.max_images is None
    if use_zip:
        images_zip = args.dest / "val2017.zip"
        _download_to(VAL_ZIP_URL, images_zip)
        if not image_dir.is_dir():
            print("Extracting images ...", file=sys.stderr)
            with zipfile.ZipFile(images_zip) as archive:
                archive.extractall(args.dest)
            (args.dest / "val2017").rename(image_dir)
        obtained = {im["file_name"] for im in images if (image_dir / im["file_name"]).is_file()}
    else:
        print(
            f"Downloading up to "
            f"{args.max_images or '~'} images / "
            f"{f'{args.max_gb} GB' if args.max_gb else 'all'} from {args.split} ...",
            file=sys.stderr,
        )
        obtained = _download_images_individually(
            [im["file_name"] for im in images], args.split, image_dir,
            args.max_images, args.max_gb,
        )

    images = [im for im in images if im["file_name"] in obtained]
    keep_ids = {im["id"] for im in images}
    annotations = [a for a in annotations if a["image_id"] in keep_ids]
    train, val = _split(images, annotations, categories, args.val_fraction)
    (args.dest / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (args.dest / "val.json").write_text(json.dumps(val), encoding="utf-8")

    config = REPO_ROOT / "configs" / "coco.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    scope = f"classes {args.classes}" if args.classes else "all 80 classes"
    config.write_text(
        f"# Auto-generated by scripts/get_coco.py. COCO {args.split} ({scope}).\n"
        "data_format: coco\n"
        f"train_source: {(args.dest / 'train.json').relative_to(REPO_ROOT)}\n"
        f"val_source: {(args.dest / 'val.json').relative_to(REPO_ROOT)}\n"
        f"image_root: {image_dir.relative_to(REPO_ROOT)}\n",
        encoding="utf-8",
    )
    print(
        f"COCO ready ({args.split}, {scope}): {len(train['images'])} train / "
        f"{len(val['images'])} val images, {len(categories)} classes at {args.dest}\n"
        f"Config written to {config}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
