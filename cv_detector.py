#!/usr/bin/env python3
"""Live detection with a trained LOFOP model -- webcam, video file, or image.

Opens a camera (or a video/image), runs the LOFOP detector on each frame, draws
the predicted boxes, and overlays the live FPS. Press ``q`` to quit.

    # webcam (device 0), using the weights the accuracy benchmark just trained:
    python cv_detector.py --data-config configs/coco128.yaml

    # a video file, saving the annotated output:
    python cv_detector.py --source clip.mp4 --data-config configs/coco128.yaml --save out.mp4

    # a single image:
    python cv_detector.py --source photo.jpg --data-config configs/coco128.yaml

Needs OpenCV: ``pip install -r requirements-webcam.txt``. Drawing is done with
OpenCV directly (fast, no per-frame image conversions beyond color order).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = REPO_ROOT / "results" / "checkpoints" / "best.pt"

# A fixed BGR palette so a class keeps its color across frames.
_PALETTE = [
    (76, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245), (180, 30, 145),
    (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250), (36, 99, 154),
]


def _class_info(data_config: Path | None, num_classes: int) -> tuple[list[str], int]:
    if data_config is None:
        return [f"class_{i}" for i in range(num_classes)], num_classes
    import yaml

    from lofop.data import load_dataset

    cfg = yaml.safe_load(Path(data_config).read_text(encoding="utf-8"))
    kwargs = {"image_root": cfg["image_root"]} if cfg.get("image_root") else {}
    dataset = load_dataset(cfg["data_format"], cfg.get("val_source") or cfg["train_source"], **kwargs)
    names = [c.name for c in dataset.categories]
    return names, len(names)


def _draw(frame, detections, class_names) -> None:
    import cv2

    for (x1, y1, x2, y2), score, label in zip(
        detections.boxes, detections.scores, detections.labels
    ):
        color = _PALETTE[label % len(_PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        caption = f"{class_names[label]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, caption, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _overlay_status(frame, fps: float, count: int, score_thr: float, nms_iou: float) -> None:
    import cv2

    text = f"FPS {fps:5.1f}  objects {count}  conf {score_thr:.2f}  iou {nms_iou:.2f}"
    keys = "+/- conf   [/] iou   q quit"
    cv2.rectangle(frame, (0, 0), (430, 44), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, keys, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError:
        print("OpenCV is required: pip install -r requirements-webcam.txt", file=sys.stderr)
        return 1
    from PIL import Image

    from lofop import Detector

    if not Path(args.checkpoint).is_file():
        print(
            f"No checkpoint at {args.checkpoint}. Train first, e.g.\n"
            "  python run_benchmarks.py --data-config configs/coco128.yaml "
            "--variant n --epochs 100 --acc-size 640 --skip-latency --skip-structural",
            file=sys.stderr,
        )
        return 1

    class_names, num_classes = _class_info(args.data_config, args.num_classes)
    detector = Detector(
        args.variant, num_classes=num_classes, checkpoint=args.checkpoint,
        class_names=class_names, image_size=args.size,
    )

    source: int | str = int(args.source) if str(args.source).isdigit() else args.source
    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp",
    }

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"Could not read image {source}", file=sys.stderr)
            return 1
        return _run_single_image(cv2, detector, frame, class_names, args)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(
            f"Could not open source {source!r}. For a webcam try --source 0 (or 1). "
            "On a laptop, make sure no other app is using the camera.",
            file=sys.stderr,
        )
        return 1

    writer = None
    smoothed = 0.0
    score_thr = args.score_threshold
    nms_iou = float(detector.model.nms_iou)
    print(
        "Running live detection. Keys: +/- confidence threshold, [/] NMS IoU, q quit.",
        file=sys.stderr,
    )
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        start = time.perf_counter()
        detector.model.nms_iou = nms_iou
        detections = detector.predict(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            score_threshold=score_thr,
        )[0]
        _draw(frame, detections, class_names)
        instant = 1.0 / max(time.perf_counter() - start, 1e-6)
        smoothed = instant if smoothed == 0 else 0.9 * smoothed + 0.1 * instant
        _overlay_status(frame, smoothed, len(detections.boxes), score_thr, nms_iou)

        if args.save:
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
                )
            writer.write(frame)

        cv2.imshow("LOFOP live detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("+"), ord("=")):   # raise the confidence cut -> fewer boxes
            score_thr = min(round(score_thr + 0.05, 2), 0.95)
        elif key in (ord("-"), ord("_")):  # lower the cut -> more boxes
            score_thr = max(round(score_thr - 0.05, 2), 0.05)
        elif key == ord("]"):              # allow more overlap between kept boxes
            nms_iou = min(round(nms_iou + 0.05, 2), 0.95)
        elif key == ord("["):              # merge overlapping boxes more aggressively
            nms_iou = max(round(nms_iou - 0.05, 2), 0.10)

    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {args.save}", file=sys.stderr)
    cv2.destroyAllWindows()
    return 0


def _run_single_image(cv2, detector, frame, class_names, args) -> int:
    from PIL import Image

    detections = detector.predict(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        score_threshold=args.score_threshold,
    )[0]
    _draw(frame, detections, class_names)
    out = args.save or "detection.png"
    cv2.imwrite(out, frame)
    print(f"{len(detections.boxes)} detections; wrote {out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live LOFOP detection (webcam/video/image).")
    parser.add_argument("--source", default="0", help="webcam index (0), video file, or image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--variant", default="n")
    parser.add_argument("--data-config", type=Path, default=None,
                        help="dataset YAML; used to read class names / count")
    parser.add_argument("--num-classes", type=int, default=80,
                        help="used only when --data-config is not given")
    parser.add_argument("--size", type=int, default=640, help="inference resolution")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--save", default=None, help="write annotated output (video .mp4 / image)")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
