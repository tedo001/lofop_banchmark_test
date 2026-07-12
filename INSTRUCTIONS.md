# How to Run the LOFOP Benchmark System

A step-by-step guide for your stack: **PyCharm** (edit/run) + **GitHub** (store
code and results) + **Claude Code** (iterate), on your **RTX 4060 OMEN laptop**
(CUDA GPU).

The committed numbers in `results/` are a CPU reference. Run this on your 4060 to
get real GPU latency, throughput, VRAM, and much faster accuracy training.

---

## 0. What this measures

| Benchmark | Output | Needs a GPU? |
|---|---|---|
| **environment** | the exact machine + versions | no |
| **structural** | params, FLOPs, model size, forward FPS | no (GPU adds GPU FPS) |
| **latency** | p50/p90/p99 latency, throughput per batch size, peak VRAM | GPU strongly recommended |
| **accuracy** | mAP@50, mAP@50:95, precision, recall, F1 | works on CPU; GPU is far faster |

Everything writes `results/*.md`, `*.csv`, `*.json` (+ optional `*.png` charts).

---

## 1. One-time setup in PyCharm

1. **Clone the repo.** PyCharm -> *Get from VCS* ->
   `https://github.com/tedo001/lofop_banchmark_test` -> Clone.
2. **Create a project virtualenv.** *Settings -> Project -> Python Interpreter ->
   Add Interpreter -> Virtualenv (new)*, Python 3.10 or 3.11. PyCharm makes
   `.venv/` (already git-ignored).
3. **Open the built-in Terminal** (Alt+F12) — it auto-activates `.venv`.

### Install PyTorch with CUDA (the important step for your 4060)

Your RTX 4060 needs the CUDA build of PyTorch, **not** the default CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the harness (this pulls in LOFOP from GitHub):

```bash
pip install -r requirements.txt
pip install -r requirements-plots.txt   # optional: charts
```

### Verify the GPU is visible

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce RTX 4060 ...
lofop doctor                                  # LOFOP's own environment check
```

If it prints `False`, update your NVIDIA driver and reinstall the `cu121` wheel.

---

## 2. Run a benchmark

Use the PyCharm terminal (or make a *Run Configuration* for `run_benchmarks.py`).

### Quick GPU speed check (no training, ~1 minute)

```bash
python run_benchmarks.py --device cuda --amp --skip-accuracy --plots
```

Gives you structural metrics + GPU latency/throughput/VRAM for all three
variants, at FP16 (`--amp`), plus charts in `results/`.

### Full run including accuracy

On synthetic data (deterministic, no dataset needed):

```bash
python run_benchmarks.py --device cuda --variant s --epochs 50 --acc-size 640 --plots
```

On **your own COCO/YOLO/VOC dataset** (real mAP):

```bash
cp configs/coco.example.yaml configs/my_data.yaml
# edit configs/my_data.yaml with your paths, then:
python run_benchmarks.py --device cuda --data-config configs/my_data.yaml \
    --variant s --epochs 100 --acc-size 640 --plots
```

### Useful flags

| Flag | Meaning |
|---|---|
| `--device cuda\|cpu\|auto` | where to run (`auto` picks the GPU if present) |
| `--amp` | FP16 mixed precision for the latency benchmark (faster on RTX) |
| `--size 640` | resolution for structural + latency |
| `--batch-sizes 1,4,8,16` | batch sizes for the latency/throughput sweep |
| `--variant n\|s\|ex` | which model the accuracy run trains |
| `--epochs 100` | accuracy training length |
| `--acc-size 640` | accuracy training/eval resolution |
| `--data-config FILE` | dataset YAML (COCO/YOLO/VOC) for real accuracy |
| `--plots` | render PNG charts (needs matplotlib) |
| `--skip-structural / --skip-latency / --skip-accuracy` | run a subset |

> **4060 tip (8 GB VRAM):** start with `--batch-sizes 1,4,8,16 --amp`. If you hit
> a CUDA out-of-memory error on `ex` at 640px, drop to `--batch-sizes 1,4,8` or
> `--size 512`. `results/latency.md` reports peak VRAM so you can see headroom.

---

## 3. Save results back to GitHub (in PyCharm)

The `results/` files are meant to be committed so your GPU numbers live in the
repo history.

1. PyCharm *Commit* tab (Ctrl+K): stage the changed `results/*` files.
2. Message e.g. `RTX 4060 benchmark run`.
3. *Push* (Ctrl+Shift+K).

Now `README.md` / `results/` show your real hardware numbers on GitHub.

---

## 4. Iterate with Claude Code

From the repo root (or the Claude Code panel in your IDE), ask for changes like:

- "Add a batch-size 32 column and re-run the latency benchmark on cuda."
- "Point the accuracy benchmark at my VOC dataset in `configs/my_data.yaml`."
- "Add a chart comparing FP16 vs FP32 latency."
- "The `ex` variant OOMs at 640 — add an automatic batch-size fallback."

Claude Code edits the harness, you run it on the 4060 in PyCharm, then commit the
new `results/` — a tight edit -> measure -> commit loop across the three tools.

---

## 5. Continuous benchmarking (optional)

`.github/workflows/benchmark.yml` runs the suite on every push and weekly on
GitHub's CPU runners (structural + synthetic accuracy), uploading `results/` as
an artifact. GitHub-hosted runners have no GPU, so treat CI as a *regression
guard* and your 4060 as the source of real speed/accuracy numbers.
