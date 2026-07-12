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

```bash
# On an RTX GPU, install the CUDA build of PyTorch first:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-plots.txt   # optional charts
```

## Run

```bash
# GPU speed check, FP16, with charts (no training):
python run_benchmarks.py --device cuda --amp --skip-accuracy --plots

# Full run on synthetic data:
python run_benchmarks.py --device cuda --variant s --epochs 50 --acc-size 640 --plots

# Accuracy on your own COCO/YOLO/VOC dataset:
cp configs/coco.example.yaml configs/my_data.yaml   # edit paths
python run_benchmarks.py --device cuda --data-config configs/my_data.yaml \
    --variant s --epochs 100 --acc-size 640
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for every flag and 4060-specific tips
(VRAM, batch sizes, OOM fallback).

## Reference results (CPU baseline — regenerate on your GPU)

Committed under `results/`. These are a CPU reference so the repo has real data;
your 4060 run will add GPU FPS, VRAM, and much lower latency.

**Structural (640x640, batch 1, CPU):**

| | n | s | ex |
|---|--:|--:|--:|
| Parameters | 1,306,129 | 3,844,297 | 20,120,785 |
| FLOPs | 6.54 G | 15.93 G | 83.24 G |
| Model Size | 5.3 MB | 15.4 MB | 80.7 MB |
| CPU FPS | 22.9 | 11.5 | 2.8 |

**Latency p50 per image (640px, CPU):** n 43.2 ms (23 FPS) · s 83.6 ms (12 FPS) ·
ex 317 ms (3 FPS) at batch 1. On an RTX 4060 with `--amp` expect these to drop by
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
configs/          # dataset config template for the accuracy benchmark
results/          # committed md/CSV/JSON/PNG from the latest run
.github/workflows/benchmark.yml  # CI regression guard (CPU)
```

## Continuous benchmarking

`.github/workflows/benchmark.yml` runs the suite on push and weekly on GitHub's
CPU runners, uploading `results/` as an artifact. CI is the regression guard;
your 4060 is the source of real speed/accuracy numbers.
