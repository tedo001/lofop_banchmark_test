#!/usr/bin/env python3
"""Run the LOFOP benchmark suite and write results to ``results/``.

Examples::

    # Full suite, auto-detecting the GPU (RTX 4060 etc.):
    python run_benchmarks.py --device auto

    # Structural + GPU latency only (no training), with mixed precision:
    python run_benchmarks.py --device cuda --amp --skip-accuracy

    # Real-dataset accuracy on COCO, GPU, 50 epochs, and charts:
    python run_benchmarks.py --device cuda --plots \
        --data-format coco --train-source data/train.json --val-source data/val.json \
        --image-root data/images --variant s --epochs 50 --acc-size 640

A dataset config file (see configs/coco.example.yaml) can supply the --data-*
values instead of passing them on the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from lofop_bench import run_accuracy, run_latency, run_structural, write_environment
from lofop.utils import render_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_data_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(","))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LOFOP benchmark harness.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--size", type=int, default=640, help="structural/latency resolution")
    parser.add_argument("--batch-sizes", type=_int_list, default=(1, 4, 8),
                        help="comma-separated batch sizes for the latency benchmark")
    parser.add_argument("--amp", action="store_true", help="mixed precision for latency (CUDA)")
    parser.add_argument("--plots", action="store_true", help="render PNG charts (needs matplotlib)")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    # Accuracy run.
    parser.add_argument("--variant", default="n", help="variant for the accuracy run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--acc-size", type=int, default=96, help="accuracy training resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="accuracy training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="accuracy training learning rate")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader workers; raise (e.g. 8) to speed up loading on big datasets",
    )
    parser.add_argument("--limit-train-images", type=int, default=None,
                        help="train on only the first N images (faster epochs)")
    parser.add_argument("--limit-val-images", type=int, default=None,
                        help="evaluate on only the first N images")
    parser.add_argument("--data-config", type=Path, default=None,
                        help="YAML with data_format/train_source/val_source/image_root")
    parser.add_argument("--data-format", default=None)
    parser.add_argument("--train-source", default=None)
    parser.add_argument("--val-source", default=None)
    parser.add_argument("--image-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_cfg = _load_data_config(args.data_config)
    data_format = args.data_format or data_cfg.get("data_format")
    train_source = args.train_source or data_cfg.get("train_source")
    val_source = args.val_source or data_cfg.get("val_source")
    image_root = args.image_root or data_cfg.get("image_root")

    env = write_environment(args.results_dir)
    print(env.to_markdown(), file=sys.stderr)
    device = "cuda" if (args.device == "auto" and env.cuda_available) else args.device
    if device == "auto":
        device = "cpu"

    structural_reports = None
    latency_results = None
    accuracy_result = None

    if not args.skip_structural:
        print("== Structural benchmark ==", file=sys.stderr)
        structural_reports = run_structural(args.results_dir, image_size=args.size)
        print(render_table(structural_reports))

    if not args.skip_latency:
        print(f"== Latency benchmark ({device}, amp={args.amp}) ==", file=sys.stderr)
        latency_results = run_latency(
            args.results_dir, device=device, image_size=args.size,
            batch_sizes=args.batch_sizes, amp=args.amp,
        )
        from lofop_bench.latency import render_latency_markdown
        print(render_latency_markdown(latency_results))

    if not args.skip_accuracy:
        print(f"== Accuracy benchmark ({device}, training) ==", file=sys.stderr)
        accuracy_result = run_accuracy(
            args.results_dir, variant=args.variant, device=device, epochs=args.epochs,
            image_size=args.acc_size, batch_size=args.batch_size, lr=args.lr,
            workers=args.workers, data_format=data_format, train_source=train_source,
            val_source=val_source, image_root=image_root,
            limit_train=args.limit_train_images, limit_val=args.limit_val_images,
        )
        print(accuracy_result.to_markdown())

    if args.plots:
        print("== Rendering charts ==", file=sys.stderr)
        from lofop_bench.plots import (
            plot_accuracy_curve,
            plot_latency_by_batch,
            plot_size_vs_speed,
            render_summary_image,
        )
        if structural_reports:
            print(f"  {plot_size_vs_speed(structural_reports, args.results_dir)}", file=sys.stderr)
        if latency_results:
            print(f"  {plot_latency_by_batch(latency_results, args.results_dir)}", file=sys.stderr)
        if accuracy_result is not None:
            history_file = args.results_dir / "accuracy_history.json"
            if history_file.is_file():
                import json as _json

                curve = plot_accuracy_curve(
                    _json.loads(history_file.read_text(encoding="utf-8")), args.results_dir
                )
                if curve:
                    print(f"  {curve}  <- mAP-vs-epoch (still rising? train longer)",
                          file=sys.stderr)
        summary = render_summary_image(
            args.results_dir, environment=env, structural=structural_reports,
            latency=latency_results, accuracy=accuracy_result,
        )
        print(f"  {summary}  <- single results image", file=sys.stderr)

    print(f"Results written to {args.results_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
