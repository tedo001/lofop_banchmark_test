#!/usr/bin/env python3
"""LOFOP Detection Studio -- a small Tkinter app for photos and videos.

Load a local photo or video, run the trained LOFOP model on it, watch the
annotated preview, and get a saved report (per-class counts, confidences,
processing speed) plus the annotated media under results/app_reports/.

    python lofop-app.py

Uses the weights from results/models/ (newest .pt) or results/checkpoints/
best.pt, and class names from configs/pipeline.yaml. Both can be changed in
the app. Needs OpenCV (pip install -r requirements-webcam.txt); tkinter ships
with the standard Windows Python installer.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # headless boxes; the analysis core still works
    tk = None

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"
REPORT_ROOT = ROOT / "results" / "app_reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

_PALETTE = [
    (76, 25, 230), (75, 180, 60), (216, 99, 67), (49, 130, 245), (180, 30, 145),
    (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250), (36, 99, 154),
]


def find_default_checkpoint() -> Path | None:
    models = sorted((ROOT / "results" / "models").glob("*.pt"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if models:
        return models[0]
    best = ROOT / "results" / "checkpoints" / "best.pt"
    return best if best.is_file() else None


def load_class_names(config_path: Path | None) -> list[str]:
    """Class names from a dataset config; falls back to a 1-class person model."""
    if config_path and Path(config_path).is_file():
        import yaml

        from lofop.data import load_dataset

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        kwargs = {"image_root": ROOT / cfg["image_root"]} if cfg.get("image_root") else {}
        dataset = load_dataset(
            cfg["data_format"], ROOT / (cfg.get("val_source") or cfg["train_source"]), **kwargs
        )
        return [c.name for c in dataset.categories]
    return ["person"]


def draw_detections(frame, detections, class_names) -> None:
    """Draw boxes + captions on a BGR frame in place."""
    import cv2

    for (x1, y1, x2, y2), score, label in zip(
        detections.boxes, detections.scores, detections.labels
    ):
        color = _PALETTE[label % len(_PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        name = class_names[label] if label < len(class_names) else str(label)
        caption = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(frame, caption, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class Analyzer:
    """Loads the detector once and analyzes photos/videos, collecting stats."""

    def __init__(self) -> None:
        self.detector = None
        self.class_names: list[str] = []

    def load(self, checkpoint: Path, config_path: Path | None, image_size: int) -> None:
        from lofop import Detector

        self.class_names = load_class_names(config_path)
        self.detector = Detector(
            "n", num_classes=len(self.class_names), checkpoint=checkpoint,
            class_names=self.class_names, image_size=image_size,
        )

    def detect_frame(self, frame_bgr, confidence: float):
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.detector.predict(Image.fromarray(rgb), score_threshold=confidence)[0]


def _new_stats() -> dict:
    return {"frames": 0, "detections": 0, "per_class": {}, "confidence_sum": 0.0,
            "peak_per_frame": 0, "started": time.perf_counter()}


def _update_stats(stats: dict, detections, class_names) -> None:
    stats["frames"] += 1
    count = len(detections.boxes)
    stats["detections"] += count
    stats["peak_per_frame"] = max(stats["peak_per_frame"], count)
    for score, label in zip(detections.scores, detections.labels):
        name = class_names[label] if label < len(class_names) else str(label)
        stats["per_class"][name] = stats["per_class"].get(name, 0) + 1
        stats["confidence_sum"] += score


def build_report(source: Path, kind: str, stats: dict, settings: dict,
                 stopped_early: bool = False) -> dict:
    elapsed = time.perf_counter() - stats["started"]
    mean_conf = stats["confidence_sum"] / stats["detections"] if stats["detections"] else 0.0
    return {
        "source": str(source),
        "kind": kind,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
        "frames_analyzed": stats["frames"],
        "total_detections": stats["detections"],
        "detections_per_frame": round(stats["detections"] / max(stats["frames"], 1), 2),
        "peak_detections_in_a_frame": stats["peak_per_frame"],
        "mean_confidence": round(mean_conf, 4),
        "per_class_counts": dict(sorted(stats["per_class"].items(), key=lambda kv: -kv[1])),
        "processing_seconds": round(elapsed, 2),
        "processing_fps": round(stats["frames"] / elapsed, 2) if elapsed > 0 else 0.0,
        "stopped_early": stopped_early,
    }


def report_markdown(report: dict) -> str:
    lines = [
        "# LOFOP detection report",
        "",
        f"- Source     : {report['source']}  ({report['kind']})",
        f"- Analyzed   : {report['analyzed_at']}",
        f"- Model      : {report['settings']['checkpoint']}",
        f"- Confidence : {report['settings']['confidence']}   "
        f"Size: {report['settings']['image_size']}px",
        "",
        f"| Frames analyzed | {report['frames_analyzed']} |",
        "|---|---|",
        f"| Total detections | {report['total_detections']} |",
        f"| Detections / frame | {report['detections_per_frame']} |",
        f"| Peak in one frame | {report['peak_detections_in_a_frame']} |",
        f"| Mean confidence | {report['mean_confidence']} |",
        f"| Processing speed | {report['processing_fps']} FPS |",
        "",
        "## Detections by class",
        "",
    ]
    if report["per_class_counts"]:
        lines += [f"- {name}: {count}" for name, count in report["per_class_counts"].items()]
    else:
        lines.append("- (none above the confidence threshold)")
    if report.get("stopped_early"):
        lines += ["", "Note: analysis was stopped before the end of the video."]
    return "\n".join(lines) + "\n"


def _new_report_dir() -> Path:
    """A unique per-run report directory; suffixes on collision."""
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        out_dir = REPORT_ROOT / (stamp if suffix == 0 else f"{stamp}_{suffix:02d}")
        try:
            out_dir.mkdir(exist_ok=False)
            return out_dir
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a report directory")


def save_report(report: dict) -> Path:
    out_dir = _new_report_dir()
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return out_dir


def analyze_photo(analyzer: Analyzer, path: Path, confidence: float,
                  settings: dict, out_dir: Path | None = None):
    """Detect on one photo; returns (annotated_bgr, report, out_dir)."""
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image {path}")
    stats = _new_stats()
    detections = analyzer.detect_frame(frame, confidence)
    _update_stats(stats, detections, analyzer.class_names)
    draw_detections(frame, detections, analyzer.class_names)
    report = build_report(path, "photo", stats, settings)
    saved = out_dir or save_report(report)
    cv2.imwrite(str(saved / f"{path.stem}_annotated.png"), frame)
    if out_dir:  # caller manages report files
        (saved / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (saved / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return frame, report, saved


def analyze_video(analyzer: Analyzer, path: Path, confidence: float, frame_step: int,
                  settings: dict, on_frame, stop_event: threading.Event):
    """Detect over a video; returns (report, out_dir).

    ``on_frame(annotated_bgr, index, total)`` is called for each processed
    frame (the GUI uses it for the live preview). Honors ``stop_event``.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    out_dir = _new_report_dir()
    writer = None
    stats = _new_stats()
    index = 0
    stopped = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if stop_event.is_set():
            stopped = True
            break
        if index % max(frame_step, 1) == 0:
            detections = analyzer.detect_frame(frame, confidence)
            _update_stats(stats, detections, analyzer.class_names)
            draw_detections(frame, detections, analyzer.class_names)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_dir / f"{path.stem}_annotated.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(fps / max(frame_step, 1), 1.0), (width, height),
                )
            writer.write(frame)
            on_frame(frame, index, total)
        index += 1
    capture.release()
    if writer is not None:
        writer.release()
    report = build_report(path, "video", stats, settings, stopped_early=stopped)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return report, out_dir


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App:
    PREVIEW_W, PREVIEW_H = 760, 460

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("LOFOP Detection Studio")
        self.analyzer = Analyzer()
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._preview_image = None  # keep a reference or Tk drops it

        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="Open Photo...", command=self.open_photo).pack(side="left")
        ttk.Button(controls, text="Open Video...", command=self.open_video).pack(
            side="left", padx=6)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Label(controls, text="Confidence").pack(side="left", padx=(18, 4))
        self.confidence = tk.DoubleVar(value=0.30)
        ttk.Scale(controls, from_=0.05, to=0.90, variable=self.confidence,
                  length=140).pack(side="left")
        self.conf_label = ttk.Label(controls, text="0.30", width=5)
        self.conf_label.pack(side="left")
        self.confidence.trace_add(
            "write", lambda *_: self.conf_label.config(text=f"{self.confidence.get():.2f}"))
        ttk.Label(controls, text="Frame step").pack(side="left", padx=(18, 4))
        self.frame_step = tk.IntVar(value=2)
        ttk.Spinbox(controls, from_=1, to=10, textvariable=self.frame_step,
                    width=4).pack(side="left")

        paths = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        paths.pack(fill="x")
        checkpoint = find_default_checkpoint()
        self.checkpoint_var = tk.StringVar(value=str(checkpoint) if checkpoint else "")
        self.config_var = tk.StringVar(
            value=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else "")
        ttk.Label(paths, text="Model:").pack(side="left")
        ttk.Entry(paths, textvariable=self.checkpoint_var, width=52).pack(side="left", padx=4)
        ttk.Button(paths, text="...", width=3, command=self.pick_checkpoint).pack(side="left")
        ttk.Label(paths, text="Config:").pack(side="left", padx=(12, 0))
        ttk.Entry(paths, textvariable=self.config_var, width=34).pack(side="left", padx=4)
        ttk.Button(paths, text="...", width=3, command=self.pick_config).pack(side="left")

        self.preview = ttk.Label(
            self.root, text="\nOpen a photo or a video to analyze\n", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=8, pady=4)
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=8)

        report_frame = ttk.Frame(self.root, padding=8)
        report_frame.pack(fill="both")
        self.report_text = tk.Text(report_frame, height=12, width=100, state="disabled")
        self.report_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(report_frame, command=self.report_text.yview)
        scroll.pack(side="right", fill="y")
        self.report_text.config(yscrollcommand=scroll.set)

        self.status = ttk.Label(self.root, text="Ready", padding=(8, 2))
        self.status.pack(fill="x")
        self.root.after(40, self.poll)

    # -- actions -----------------------------------------------------------

    def pick_checkpoint(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("Model weights", "*.pt")])
        if chosen:
            self.checkpoint_var.set(chosen)

    def pick_config(self) -> None:
        chosen = filedialog.askopenfilename(filetypes=[("Dataset config", "*.yaml *.yml")])
        if chosen:
            self.config_var.set(chosen)

    def _settings(self) -> dict:
        return {
            "checkpoint": self.checkpoint_var.get(),
            "config": self.config_var.get() or None,
            "confidence": round(self.confidence.get(), 2),
            "image_size": 416,
            "frame_step": int(self.frame_step.get()),
        }

    def _ensure_model(self) -> bool:
        checkpoint = self.checkpoint_var.get()
        if not checkpoint or not Path(checkpoint).is_file():
            messagebox.showerror(
                "No model", "No trained weights found.\nTrain first "
                "(python pipeline.py --device cuda --fast) or pick a .pt file.")
            return False
        if self.analyzer.detector is None:
            self.set_status("Loading model (first time takes a few seconds)...")
            self.root.update_idletasks()
            config = Path(self.config_var.get()) if self.config_var.get() else None
            try:
                self.analyzer.load(Path(checkpoint), config, image_size=416)
            except Exception as exc:  # bad checkpoint/config combo etc.
                messagebox.showerror("Model load failed", str(exc))
                return False
        return True

    def open_photo(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images", " ".join(f"*{e}" for e in sorted(IMAGE_EXTS)))])
        if not path or not self._ensure_model():
            return
        self.start_worker(self._photo_worker, Path(path))

    def open_video(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        path = filedialog.askopenfilename(
            filetypes=[("Videos", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS)))])
        if not path or not self._ensure_model():
            return
        self.start_worker(self._video_worker, Path(path))

    def start_worker(self, target, path: Path) -> None:
        self.stop_event.clear()
        self.stop_button.config(state="normal")
        self.set_status(f"Analyzing {path.name} ...")
        self.worker = threading.Thread(target=target, args=(path,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()

    # -- workers (background threads; communicate via the queue) -------------

    def _photo_worker(self, path: Path) -> None:
        try:
            frame, report, out_dir = analyze_photo(
                self.analyzer, path, self.confidence.get(), self._settings())
            self.queue.put(("frame", frame, 1, 1))
            self.queue.put(("done", report, out_dir))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    def _video_worker(self, path: Path) -> None:
        try:
            report, out_dir = analyze_video(
                self.analyzer, path, self.confidence.get(), int(self.frame_step.get()),
                self._settings(),
                on_frame=lambda f, i, t: self.queue.put(("frame", f, i, t)),
                stop_event=self.stop_event,
            )
            self.queue.put(("done", report, out_dir))
        except Exception as exc:
            self.queue.put(("error", str(exc)))

    # -- UI updates ----------------------------------------------------------

    def poll(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "frame":
                    _, frame, index, total = item
                    self.show_frame(frame)
                    if total:
                        self.progress["value"] = 100.0 * index / total
                elif item[0] == "done":
                    _, report, out_dir = item
                    self.progress["value"] = 100
                    self.stop_button.config(state="disabled")
                    self.show_report(report, out_dir)
                elif item[0] == "error":
                    self.stop_button.config(state="disabled")
                    self.set_status("Error")
                    messagebox.showerror("Analysis failed", item[1])
        except queue.Empty:
            pass
        self.root.after(40, self.poll)

    def show_frame(self, frame_bgr) -> None:
        import cv2
        from PIL import Image, ImageTk

        height, width = frame_bgr.shape[:2]
        scale = min(self.PREVIEW_W / width, self.PREVIEW_H / height, 1.0)
        resized = cv2.resize(frame_bgr, (int(width * scale), int(height * scale)))
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self._preview_image = ImageTk.PhotoImage(image)
        self.preview.config(image=self._preview_image, text="")

    def show_report(self, report: dict, out_dir: Path) -> None:
        text = report_markdown(report) + f"\nSaved to: {out_dir}\n"
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", text)
        self.report_text.config(state="disabled")
        self.set_status(f"Done - report saved to {out_dir}")

    def set_status(self, text: str) -> None:
        self.status.config(text=text)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    if tk is None:
        print("tkinter is not available in this Python. On Windows, install Python "
              "from python.org (tkinter is included).", file=sys.stderr)
        return 1
    App().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
