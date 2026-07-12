# LOFOP Benchmark Harness

Reproducible benchmarks for the [LOFOP](https://github.com/tedo001/LOFOP)
computer-vision framework. This repository installs LOFOP and drives it through
its **public API only**, so it doubles as an integration test of the released
package and as a regression guard for size, speed, and accuracy.

Two benchmark families:

- **Structural** - parameters, FLOPs, state size, CPU (and GPU when available)
  forward FPS, and peak memory for every `lofop-detect` variant. No training.
- **Accuracy** - a short, fully deterministic train-and-evaluate run on LOFOP's
  built-in synthetic `shapes` dataset, reporting mAP / precision / recall / F1.
  A controlled instrument for catching quality regressions between LOFOP
  versions, not a real-world accuracy claim.

Each writes markdown, CSV, and JSON to `results/` so a run is readable by both
humans and CI.

## Install

```bash
python -m pip install -r requirements.txt
```

By default this pulls LOFOP from its `tedo` development branch; pin a released
version in `requirements.txt` once LOFOP is on PyPI.

## Run

```bash
python run_benchmarks.py                     # structural (640px) + accuracy (30 epochs)
python run_benchmarks.py --skip-accuracy     # structural only (fast)
python run_benchmarks.py --size 640 --epochs 30 --acc-size 96
```

## Latest results

Measured on CPU; committed under `results/`. Regenerate with `run_benchmarks.py`
(numbers vary with hardware; the accuracy run is deterministic for a fixed
seed/epochs/size).

### Structural (640x640, batch 1, CPU)

| Metric | lofop-detect-n | lofop-detect-s | lofop-detect-ex |
|---|---:|---:|---:|
| Parameters | 1,306,129 | 3,844,297 | 20,120,785 |
| FLOPs | 6.54 G | 15.93 G | 83.24 G |
| Model Size | 5.3 MB | 15.4 MB | 80.7 MB |
| CPU FPS | 24.1 | 11.5 | 2.9 |

### Accuracy (synthetic `shapes`, lofop-detect-n, 30 CPU epochs @ 96px, seed 0)

| Metric | Value |
|---|---:|
| mAP@50 | 0.5731 |
| mAP@50:95 | 0.3418 |
| Precision | 0.3881 |
| Recall | 0.8125 |
| F1 | 0.5253 |

## Layout

```
lofop_bench/
  structural.py   # size/speed of every variant (reuses lofop.utils.benchmark)
  accuracy.py     # deterministic train + evaluate on synthetic shapes
run_benchmarks.py # CLI entry point -> results/
results/          # committed md/CSV/JSON from the latest run
.github/workflows/benchmark.yml  # runs the suite in CI
```

## Continuous benchmarking

`.github/workflows/benchmark.yml` runs the harness on every push and on a weekly
schedule, uploading the `results/` artifacts so size/speed/accuracy drift is
visible over time.
