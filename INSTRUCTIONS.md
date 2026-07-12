# LOFOP Benchmark — Step-by-Step (Small COCO First, Then Big)

A complete guide for your stack: **PyCharm** (edit/run) + **GitHub** (store code
and results) + **Claude Code** (iterate), on your **RTX 4060 OMEN laptop**.

The plan, in order:

1. Set up the project and GPU.
2. **Start small** — download **COCO128** (128 images, ~6 MB), train, evaluate,
   and *see the detections*.
3. **Scale up** — point the same commands at the full COCO dataset.

Do Stage A end-to-end first. Only move to Stage B once you can see boxes on
COCO128 images.

---

## 1. One-time setup (PyCharm)

1. **Clone.** PyCharm → *Get from VCS* →
   `https://github.com/tedo001/lofop_banchmark_test`.
2. **Virtualenv.** *Settings → Project → Python Interpreter → Add → Virtualenv
   (new)*, Python 3.10/3.11. PyCharm creates `.venv/` (git-ignored).
3. **Terminal.** Open PyCharm's terminal (Alt+F12); `.venv` auto-activates.

### Install CUDA PyTorch for the 4060 (do this before `requirements.txt`)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # installs the published lofop from PyPI
pip install -r requirements-plots.txt    # optional charts
```

`requirements.txt` pins `lofop[models]>=0.1.4` from
[PyPI](https://pypi.org/project/lofop/), so you are benchmarking the released
package (`pip install lofop`). To benchmark a specific version, edit the pin
(e.g. `lofop[models]==0.1.4`).

### Verify the GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True NVIDIA GeForce RTX 4060 ...
lofop doctor
```

If it prints `False`: update the NVIDIA driver, then reinstall the `cu121` wheel.

---

## Stage A — Start small with COCO128

### A1. Download the small dataset (one command)

```bash
python scripts/get_coco128.py
```

This downloads COCO128, splits it into train/val, arranges it as LOFOP YOLO
roots under `data/coco128/`, and writes `configs/coco128.yaml`. (`data/` is
git-ignored, so the dataset is not committed.)

### A2. Train + evaluate on COCO128

```bash
python run_benchmarks.py --device cuda \
    --data-config configs/coco128.yaml \
    --variant n --epochs 100 --acc-size 640 \
    --skip-latency --skip-structural
```

Writes `results/accuracy.md` / `.json` (mAP@50, mAP@50:95, precision, recall,
F1) and saves the trained weights to `results/checkpoints/best.pt`.

> COCO128 is tiny (128 images) — it is for **verifying the pipeline works**, not
> for high accuracy. Expect modest mAP; that is normal.

### A3. See it detect (the important part)

```bash
python scripts/detect_sample.py --data-config configs/coco128.yaml --size 640
```

This runs `results/checkpoints/best.pt` on a handful of COCO128 val images and
writes pictures with the predicted boxes drawn on them to
`results/detections/`. Open them in PyCharm — this is your proof the detector
trained and detects.

Raise/lower `--score-threshold` (default 0.25) to show fewer/more boxes.

### A4. (Optional) full speed benchmark on the GPU

```bash
python run_benchmarks.py --device cuda --amp --skip-accuracy --plots
```

Structural metrics + GPU latency (p50/p90/p99), throughput per batch size, and
peak VRAM for all variants, with charts in `results/`.

### A5. Save results to GitHub (PyCharm)

Commit the changed `results/*.md/.json/.png` (Ctrl+K) and push (Ctrl+Shift+K).
The dataset, checkpoints, and detection images stay local (git-ignored); the
*numbers* go to GitHub.

---

## Stage B — Scale to the full COCO dataset

Once Stage A works, use the **same commands** against full COCO.

### B1. Get COCO 2017

Download from https://cocodataset.org (or use a mirror):

- `train2017.zip` (~18 GB) and/or `val2017.zip` (~1 GB)
- `annotations_trainval2017.zip` (instances JSON)

Arrange as COCO format and create `configs/my_data.yaml`:

```yaml
data_format: coco
train_source: data/coco/annotations/instances_train2017.json
val_source: data/coco/annotations/instances_val2017.json
image_root: data/coco/images
```

(Tip: start with just `val2017` for train **and** val to shake out the full
pipeline before committing to the 18 GB train set.)

### B2. Train on COCO

```bash
python run_benchmarks.py --device cuda \
    --data-config configs/my_data.yaml \
    --variant s --epochs 100 --acc-size 640 \
    --skip-latency --skip-structural
```

Move up to `--variant s` or `--variant ex` for higher accuracy on real data.

### B3. Detect on COCO images

```bash
python scripts/detect_sample.py --data-config configs/my_data.yaml --size 640 --limit 12
```

---

## RTX 4060 tips (8 GB VRAM)

- Start with defaults. If you hit **CUDA out of memory**, in order: lower
  `--acc-size` (640 → 512), reduce the model (`ex` → `s` → `n`), or (for the
  latency benchmark) shrink `--batch-sizes`.
- `results/latency.md` reports **peak VRAM** so you can see headroom.
- `--amp` (mixed precision) is faster and uses less VRAM on RTX cards; training
  already uses AMP automatically on CUDA.
- Full-COCO training is long. Do a few epochs first to confirm loss is falling,
  then leave a full run going.

---

## Iterate with Claude Code

Ask for changes in natural language, e.g.:

- "Add a batch-size 16 and 32 column to the latency benchmark."
- "Train `s` on COCO for 300 epochs and add a mAP-vs-epoch chart."
- "`ex` OOMs at 640 — add an automatic batch-size fallback."
- "Also report per-class AP for the COCO run."

Claude Code edits the harness → you run it on the 4060 in PyCharm → commit the
new `results/`. Tight edit → measure → commit loop across all three tools.

---

## Command cheat-sheet

```bash
python scripts/get_coco128.py                                   # get small dataset
python run_benchmarks.py --device cuda --data-config configs/coco128.yaml \
    --variant n --epochs 100 --acc-size 640 --skip-latency --skip-structural   # train+eval
python scripts/detect_sample.py --data-config configs/coco128.yaml --size 640  # see boxes
python run_benchmarks.py --device cuda --amp --skip-accuracy --plots           # GPU speed
```

CI (`.github/workflows/benchmark.yml`) runs a fast CPU subset on push/weekly as
a regression guard; your 4060 is the source of real speed/accuracy numbers.
