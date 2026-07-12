#!/usr/bin/env python3
"""Run the LOFOP benchmark suite and write results to ``results/``.

Usage::

    python run_benchmarks.py                       # structural + accuracy
    python run_benchmarks.py --skip-accuracy       # structural only (fast)
    python run_benchmarks.py --size 640 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lofop_bench import run_accuracy, run_structural
from lofop.utils import render_table

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LOFOP benchmark harness.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--size", type=int, default=640, help="structural benchmark resolution")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--variant", default="n", help="variant for the accuracy run")
    parser.add_argument("--epochs", type=int, default=20, help="accuracy training epochs")
    parser.add_argument("--acc-size", type=int, default=96, help="accuracy benchmark resolution")
    args = parser.parse_args(argv)

    if not args.skip_structural:
        print("== Structural benchmark ==", file=sys.stderr)
        reports = run_structural(args.results_dir, image_size=args.size)
        print(render_table(reports))

    if not args.skip_accuracy:
        print("== Accuracy benchmark (training) ==", file=sys.stderr)
        result = run_accuracy(
            args.results_dir, variant=args.variant, epochs=args.epochs,
            image_size=args.acc_size,
        )
        print(result.to_markdown())

    print(f"Results written to {args.results_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
