"""LOFOP benchmark harness: reproducible structural, latency, and accuracy runs.

This package drives the LOFOP framework through its public API only -- it never
reaches into private internals -- so it doubles as an integration test of the
released package. Benchmark families:

* ``structural``: parameters, FLOPs, state size, forward FPS, peak memory for
  every LOFOP-Detect variant. No training.
* ``latency``: GPU-aware per-image latency percentiles (p50/p90/p99),
  throughput at several batch sizes, and peak VRAM (CUDA).
* ``accuracy``: train-and-evaluate -- deterministic on synthetic ``shapes``, or
  on a real COCO-format dataset -- reporting mAP/precision/recall/F1.
* ``environment``: the hardware/software the run was measured on.

Everything writes markdown, CSV, and/or JSON so results are readable by humans
and by CI.
"""

from lofop_bench.accuracy import AccuracyResult, run_accuracy
from lofop_bench.environment import Environment, capture, write_environment
from lofop_bench.latency import LatencyResult, run_latency
from lofop_bench.structural import run_structural

__all__ = [
    "run_structural",
    "run_latency",
    "LatencyResult",
    "run_accuracy",
    "AccuracyResult",
    "capture",
    "write_environment",
    "Environment",
]
