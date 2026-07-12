"""LOFOP benchmark harness: reproducible structural and accuracy benchmarks.

This package drives the LOFOP framework through its public API only -- it never
reaches into private internals -- so it doubles as an integration test of the
released package. Two benchmark families:

* ``structural``: parameters, FLOPs, state size, CPU (and GPU) forward FPS, and
  peak memory for every LOFOP-Detect variant. No training required.
* ``accuracy``: a short, fully deterministic train-and-evaluate run on the
  built-in synthetic ``shapes`` dataset, reporting mAP/precision/recall/F1.

Both write markdown, CSV, and JSON so results are readable by humans and by CI.
"""

from lofop_bench.accuracy import AccuracyResult, run_accuracy
from lofop_bench.structural import run_structural

__all__ = ["run_structural", "run_accuracy", "AccuracyResult"]
