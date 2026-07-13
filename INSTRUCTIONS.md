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

> ### Windows PowerShell users, read this first
>
> - **Use ONE virtualenv.** Do all `pip install` and all runs in the same
>   activated venv (e.g. `.venv`). If you create a second venv, it starts empty
>   and you'll get `ModuleNotFoundError` until you reinstall everything into it.
>   Verify you're in the right one: `python -c "import lofop, yaml, torch"` — no
>   output means good.
> - **Line continuation is a backtick `` ` ``, not `\`.** The multi-line commands
>   below use `\` (bash style). In PowerShell either replace `\` with `` ` `` or,
>   simplest, **put the whole command on one line** (the single-line versions
>   below are ready to paste).
> - **Paths use `\`** on Windows: `configs\coco128.yaml` (forward slashes also
>   work in Python, so either is fine).

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
python scripts/get_coco128.py --classes person
```

This downloads COCO128, keeps only the class(es) you name, splits it
train/val, writes a COCO-format dataset under `data/coco128/`, and generates
`configs/coco128.yaml`. (`data/` is git-ignored, so the dataset is not
committed.)

> **Why `--classes person`?** COCO128 is ~100 images. Learning **80 classes**
> from that, from scratch, is impossible — you'll get near-zero mAP. Narrowing
> to **one common class** (person appears in most images) makes it a tractable
> task, so the model actually learns and you get real detections to look at.
> Drop `--classes` to keep all 80 (a plumbing check, not a real result), or pass
> several, e.g. `--classes person car`.

### A2. Train + evaluate

```bash
python run_benchmarks.py --device cuda \
    --data-config configs/coco128.yaml \
    --variant n --epochs 100 --acc-size 640 \
    --skip-latency --skip-structural
```

**PowerShell (one line — paste this):**

```powershell
python run_benchmarks.py --device cuda --data-config configs\coco128.yaml --variant n --epochs 100 --acc-size 640 --skip-latency --skip-structural
```

A live progress bar shows the epoch, loss, and ETA while it trains. It writes
`results/accuracy.md` / `.json` (mAP@50, mAP@50:95, precision, recall, F1) and
saves the trained weights to `results/checkpoints/best.pt`.

### A3. See it detect on sample images

```bash
python scripts/detect_sample.py --data-config configs/coco128.yaml --size 640
```

Runs `results/checkpoints/best.pt` on validation images and writes pictures with
the predicted boxes drawn on them to `results/detections/`. Open them in
PyCharm. Raise/lower `--score-threshold` (default 0.25) for fewer/more boxes.

### A4. Live detection from your webcam

```bash
pip install -r requirements-webcam.txt
python cv_detector.py --data-config configs/coco128.yaml --size 640
```

Opens your laptop camera and runs the detector live with boxes + an FPS overlay.
Press **`q`** in the window to quit. Also works on a video or a single image:

```bash
python cv_detector.py --source clip.mp4  --data-config configs/coco128.yaml --save out.mp4
python cv_detector.py --source photo.jpg --data-config configs/coco128.yaml
```

### A5. Full speed benchmark + one results image

```bash
python run_benchmarks.py --device cuda --amp --plots --skip-accuracy
```

Structural metrics + GPU latency (p50/p90/p99), throughput per batch size, and
peak VRAM for all variants. With `--plots` it also writes **`results/summary.png`**
— a single report image (params, speed, latency, and accuracy in one picture)
you can drop straight into a slide or a README.

### A6. Save results to GitHub (PyCharm)

Commit the changed `results/*.md/.json/.png` (Ctrl+K) and push (Ctrl+Shift+K).
The dataset, checkpoints, and detection images stay local (git-ignored); the
*numbers* go to GitHub.

---

## Stage B — Train on real quality data (COCO val2017)

Accuracy comes from **data**: COCO128's ~100 images can't teach real detection.
Stage B trains on **5,000 real, well-annotated COCO images** — one command
downloads and prepares them, then it's the same benchmark flow.

### B1. Get COCO (one command)

**A ~4-5 GB slice of the big training set** (recommended for real accuracy):

```bash
python scripts/get_coco.py --split train2017 --max-gb 5
```

```powershell
python scripts\get_coco.py --split train2017 --max-gb 5
```

COCO2017 ships as fixed pieces — `val2017` (~1 GB, 5k images) and `train2017`
(~18 GB, 118k images); there is no official 4-5 GB split. So `--split train2017
--max-gb 5` streams images one at a time from the training set and **stops at ~5
GB** (~30,000 images), giving you a big real dataset without the full 18 GB. The
annotations (~250 MB) download once and are cached.

Options:

- `--split val2017` (default) — the 5,000-image val set as one ~1 GB zip.
- `--split train2017 --max-gb 5` — a size-capped slice of the 118k training set.
- `--max-images N` — hard cap on image count (exact).
- `--classes person car ...` — download only those classes (omit for all 80).

Examples:

```bash
python scripts/get_coco.py --split train2017 --max-gb 5                 # ~5 GB, all classes
python scripts/get_coco.py --split train2017 --classes person --max-gb 4  # ~4 GB of people
python scripts/get_coco.py                                             # full val2017 (~1 GB)
```

### B2. Train on it

```bash
python run_benchmarks.py --device cuda --data-config configs/coco.yaml --variant s --epochs 100 --acc-size 640 --skip-latency --skip-structural
```

### B3. Detect / go live on the trained model

```bash
python scripts/detect_sample.py --data-config configs/coco.yaml --size 640 --limit 12
python cv_detector.py --data-config configs/coco.yaml --size 640          # webcam
```

### Getting the accuracy up (the recipe that matters)

LOFOP ships **no pretrained weights**, so every run trains from scratch — that
takes far more epochs and data than fine-tuning. Levers, in order of impact:

1. **More images** — drop `--max-images`, or fewer `--classes`, so each class
   has more examples. This is the biggest lever.
2. **More epochs** — `--epochs 300`. From scratch needs it.
3. **Bigger model** — `--variant s` then `--variant ex` for more capacity.
4. **Full resolution** — keep `--acc-size 640`.
5. **Tune training** — `--batch-size` (raise it while VRAM allows) and `--lr`.
6. **Fewer classes** — a 1-3 class detector trained well beats a weak 80-class one.

**Read the curve to decide.** Every run with `--plots` writes
`results/accuracy_curve.png` (mAP@50 and loss vs. epoch):

- **mAP still rising at the last epoch** → train longer (raise `--epochs`).
- **mAP flat while loss still drops** → the model is saturating; go bigger
  (`--variant s`/`ex`) or add data.
- **Loss flat and mAP low** → try a different `--lr`, or you need much more data.

`results/accuracy_history.json` has the per-epoch numbers if you want to plot
them yourself.

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
- **Speed up data loading with `--workers`.** By default loading is
  single-process (slow on big datasets — a single epoch can take many minutes).
  Add `--workers 8` so images load in parallel; this often cuts epoch time
  several-fold. The run now prints **one line per epoch** (loss, mAP, epoch
  time, ETA) plus a heartbeat while an epoch is in progress, so after epoch 1
  finishes you know the total time.

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
