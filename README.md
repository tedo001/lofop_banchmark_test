# LOFOP Benchmark System

A proper, reproducible benchmark system for the
[LOFOP](https://github.com/tedo001/LOFOP) computer-vision framework. It installs
LOFOP and drives it through its **public API only**, so it doubles as an
integration test of the released package and a regression guard for size, speed,
and accuracy.

> **New here? Read [INSTRUCTIONS.md](INSTRUCTIONS.md)** for the full step-by-step
> guide (PyCharm + GitHub + Claude Code, on an RTX 4060 laptop).

## Benchmarks

| Benchmark | What it measures | GPU |
|---|---|---|
| **environment** | machine + LOFOP/PyTorch/CUDA versions (for reproducibility) | - |
| **structural** | parameters, FLOPs, model size, forward FPS | optional |
| **latency** | p50/p90/p99 per-image latency, throughput per batch size, peak VRAM | recommended |
| **accuracy** | mAP@50, mAP@50:95, precision, recall, F1 (synthetic or real dataset) | optional |

GPU timing uses CUDA events (correct for async work), and an `--amp` flag
compares FP16 vs FP32 on an RTX card. Each benchmark writes markdown, CSV/JSON,
and optional PNG charts to `results/`.

## Install

This benchmarks the **published** [`lofop`](https://pypi.org/project/lofop/)
package from PyPI (pinned in `requirements.txt`).

```bash
# On an RTX GPU, install the CUDA build of PyTorch first:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # installs lofop from PyPI
pip install -r requirements-plots.txt    # optional charts
```

The committed results were measured against `lofop 0.1.4`; `results/environment.json`
records the exact version every run used.

## Quick start: small COCO first

The recommended path — verify everything on a 128-image COCO subset, then scale
to full COCO (full walkthrough in [INSTRUCTIONS.md](INSTRUCTIONS.md)):

```bash
# 1. Get COCO128 (128 images, ~6 MB) and a ready config:
python scripts/get_coco128.py

# 2. Train + evaluate on it (GPU):
python run_benchmarks.py --device cuda --data-config configs/coco128.yaml --variant n --epochs 100 --acc-size 640 --skip-latency --skip-structural

# 3. SEE it detect - draws predicted boxes on val images -> results/detections/:
python scripts/detect_sample.py --data-config configs/coco128.yaml --size 640
```

> **Windows PowerShell:** keep each command on **one line** (don't use `\` to
> split — that's bash; PowerShell uses a backtick `` ` ``). Do every `pip install`
> and run in the **same** activated venv. See [INSTRUCTIONS.md](INSTRUCTIONS.md).

Then point the same commands at the full COCO dataset with your own
`configs/my_data.yaml`.

## Other runs

```bash
# GPU speed check, FP16, with charts (no training):
python run_benchmarks.py --device cuda --amp --skip-accuracy --plots

# Deterministic synthetic accuracy (no dataset needed):
python run_benchmarks.py --device cuda --variant s --epochs 50 --acc-size 640
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for every flag and 4060-specific tips
(VRAM, batch sizes, OOM fallback).

## Reference results (`lofop 0.1.4`, CPU baseline — regenerate on your GPU)

Committed under `results/`, measured against the published `lofop 0.1.4`. These
are a CPU reference so the repo has real data; your 4060 run will add GPU FPS,
VRAM, and much lower latency.

**Structural (640x640, batch 1, CPU):**

| | n | s | ex |
|---|--:|--:|--:|
| Parameters | 1,306,129 | 3,844,297 | 20,120,785 |
| FLOPs | 6.54 G | 15.93 G | 83.24 G |
| Model Size | 5.3 MB | 15.4 MB | 80.7 MB |
| CPU FPS | 21.8 | 10.4 | 3.0 |

**Latency p50 per image (640px, CPU):** n 44.5 ms (23 FPS) · s 86.6 ms (12 FPS) ·
ex 323 ms (3 FPS) at batch 1. On an RTX 4060 with `--amp` expect these to drop by
roughly an order of magnitude and throughput to scale with batch size.

**Accuracy (synthetic `shapes`, lofop-detect-n, 30 CPU epochs @96px, seed 0):**
mAP@50 0.573 · mAP@50:95 0.342 · P 0.388 · R 0.813 · F1 0.525 — a reproducible
regression instrument, not a real-world claim. Point `--data-config` at COCO for
real numbers.

## Layout

```
lofop_bench/
  environment.py  # capture machine + versions
  structural.py   # size/speed of every variant (reuses lofop.utils.benchmark)
  latency.py      # GPU-aware latency percentiles, throughput, VRAM
  accuracy.py     # train + evaluate: synthetic or real COCO/YOLO/VOC
  plots.py        # optional matplotlib charts
run_benchmarks.py # CLI entry point -> results/
scripts/
  get_coco128.py  # download COCO128 and arrange it for LOFOP
  detect_sample.py # run the trained detector and draw boxes on sample images
configs/          # dataset config template (+ generated coco128.yaml)
results/          # committed md/CSV/JSON/PNG from the latest run
.github/workflows/benchmark.yml  # CI regression guard (CPU)
```

## Continuous benchmarking

`.github/workflows/benchmark.yml` runs the suite on push and weekly on GitHub's
CPU runners, uploading `results/` as an artifact. CI is the regression guard;
your 4060 is the source of real speed/accuracy numbers.
