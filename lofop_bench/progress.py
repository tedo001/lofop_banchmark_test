"""Simple, dependency-free progress reporting for benchmark runs.

Prints to stderr (so it never pollutes the markdown/JSON on stdout). Two
helpers:

* :class:`ProgressBar` -- a generic ``current/total`` bar with elapsed/ETA.
* :class:`TrainingProgress` -- a context manager producing a classic training
  log during ``Detector.train``: periodic ``Training: Epoch[e/E]
  Iteration[i/N] Loss: x`` lines, a live tqdm-style overall progress line, one
  summary line per epoch, and a boxed "Training Finished" report at the end.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from typing import TextIO


def format_duration(seconds: float) -> str:
    """Human-readable duration: ``45s`` / ``6.4m`` / ``2.1h``."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class ProgressBar:
    """A minimal ``[####----] 4/10`` bar that redraws in place."""

    def __init__(
        self, total: int, *, prefix: str = "", width: int = 30, stream: TextIO | None = None,
    ) -> None:
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = width
        self.stream = stream or sys.stderr
        self._start = time.perf_counter()

    def update(self, current: int, suffix: str = "") -> None:
        fraction = min(current / self.total, 1.0)
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._start
        eta = (elapsed / fraction - elapsed) if fraction > 0 else 0.0
        self.stream.write(
            f"\r{self.prefix} [{bar}] {current}/{self.total} "
            f"{fraction * 100:5.1f}%  {suffix}  ETA {eta:4.0f}s "
        )
        self.stream.flush()

    def finish(self, suffix: str = "done") -> None:
        elapsed = time.perf_counter() - self._start
        filled = "#" * self.width
        self.stream.write(
            f"\r{self.prefix} [{filled}] {self.total}/{self.total} 100.0%  "
            f"{suffix}  {elapsed:4.0f}s elapsed\n"
        )
        self.stream.flush()


class TrainingProgress:
    """Classic training log, driven by LOFOP events plus a per-batch callback.

    Output shape (all on stderr)::

        Training: Epoch[001/020] Iteration[050/250] Loss: 2.8413
        Training: Epoch[001/020] Iteration[100/250] Loss: 2.6120
        Epoch[001/020] Train loss: 2.5107  mAP@50: 0.0123  mAP@50:95: 0.0031  time: 41.2s
        Training Progress:  5%|#-----------------------| 1/20 [41s<13.0m, 41.2s/it, loss=2.5107, mAP50=0.0123]
        ...
        =============== Training Finished ===============
        Finished Time       : 07-15_12-30
        Best mAP@50         : 0.3125
        Best Epoch          : 18
        Total Training Time : 1234.56 sec (20.58 min)
        =================================================

    Epoch summaries come from the ``train.epoch_end`` event; iteration lines
    come from :meth:`on_batch` (wired to the model's loss computation by the
    harness), so the loss shown is the real per-batch training loss. A
    background thread keeps the overall progress line alive between prints and
    flags the per-epoch validation phase.
    """

    _BAR_WIDTH = 24

    def __init__(
        self, total_epochs: int, *, prefix: str = "training", interval: float = 0.25,
        steps_per_epoch: int | None = None, log_every: int | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.total = max(int(total_epochs), 1)
        self.prefix = prefix
        self.interval = interval
        self.steps_per_epoch = steps_per_epoch
        if log_every is None:
            # Match the classic look (every 50 iterations) but stay useful on
            # small runs by logging ~5 times per short epoch.
            if steps_per_epoch and steps_per_epoch <= 50:
                log_every = max(steps_per_epoch // 5, 1)
            else:
                log_every = 50
        self.log_every = max(int(log_every), 1)
        self.stream = stream or sys.stderr
        self._subscription = None
        self._events = None
        self._start = 0.0
        self._last_epoch_time = 0.0
        self._completed = 0
        self._batches = 0
        self._last_loss: float | None = None
        self._last_map: float | None = None
        self._best_map: float | None = None
        self._best_epoch: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- wiring ------------------------------------------------------------

    def __enter__(self) -> TrainingProgress:
        from lofop.registries import EVENTS

        self._events = EVENTS
        self._start = self._last_epoch_time = time.perf_counter()
        self._subscription = EVENTS.subscribe("train.epoch_end", self._on_epoch_end)
        steps = f" x {self.steps_per_epoch} iterations" if self.steps_per_epoch else ""
        print(f"{self.prefix}: {self.total} epochs{steps}", file=self.stream)
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def on_batch(self, loss=None) -> None:
        """Record one finished training batch; log every ``log_every`` batches.

        ``loss`` may be a tensor; it is only converted (a GPU sync) on the
        batches that actually print, so overhead between logs is nil.
        """
        self._batches += 1
        at_log = self._batches % self.log_every == 0
        at_end = self.steps_per_epoch is not None and self._batches == self.steps_per_epoch
        if not (at_log or at_end):
            return
        if loss is not None:
            self._last_loss = float(loss)
        total_str = f"{self.steps_per_epoch:03d}" if self.steps_per_epoch else "???"
        loss_str = f" Loss: {self._last_loss:.4f}" if self._last_loss is not None else ""
        self._clear_line()
        print(
            f"Training: Epoch[{self._completed + 1:03d}/{self.total:03d}] "
            f"Iteration[{self._batches:03d}/{total_str}]{loss_str}",
            file=self.stream,
        )

    # -- rendering ----------------------------------------------------------

    def _clear_line(self) -> None:
        self.stream.write("\r" + " " * 110 + "\r")

    def _progress_line(self) -> str:
        now = time.perf_counter()
        elapsed = now - self._start
        if self.steps_per_epoch:
            in_epoch = min(self._batches / self.steps_per_epoch, 1.0)
        else:
            in_epoch = 0.0
        overall = min((self._completed + in_epoch) / self.total, 1.0)
        filled = int(overall * self._BAR_WIDTH)
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        if self._completed:
            per_epoch = (self._last_epoch_time - self._start) / self._completed
            remaining = max(per_epoch * (self.total - self._completed - in_epoch), 0.0)
            timing = f"[{format_duration(elapsed)}<{format_duration(remaining)}, {per_epoch:.1f}s/it"
        else:
            timing = f"[{format_duration(elapsed)}<?, ?s/it"
        postfix = ""
        if self._last_loss is not None:
            postfix += f", loss={self._last_loss:.4f}"
        if self._last_map is not None:
            postfix += f", mAP50={self._last_map:.4f}"
        evaluating = (
            "  evaluating val ..."
            if self.steps_per_epoch and self._batches >= self.steps_per_epoch
            else ""
        )
        return (
            f"Training Progress: {overall * 100:3.0f}%|{bar}| "
            f"{self._completed}/{self.total} {timing}{postfix}]{evaluating}"
        )

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            self.stream.write("\r" + self._progress_line() + "   ")
            self.stream.flush()

    # -- events --------------------------------------------------------------

    def _on_epoch_end(self, event) -> None:
        now = time.perf_counter()
        self._completed = int(event.get("epoch", 0)) + 1
        epoch_time = now - self._last_epoch_time
        self._last_epoch_time = now
        self._batches = 0
        loss = event.get("loss")
        if loss is not None:
            self._last_loss = float(loss)
        metrics = event.get("metrics") or {}
        map50 = metrics.get("map50")
        map50_95 = metrics.get("map50_95")
        if map50 is not None:
            self._last_map = float(map50)
            if self._best_map is None or self._last_map > self._best_map:
                self._best_map, self._best_epoch = self._last_map, self._completed
        parts = [f"Epoch[{self._completed:03d}/{self.total:03d}]"]
        if loss is not None:
            parts.append(f"Train loss: {loss:.4f}")
        if map50 is not None:
            parts.append(f"mAP@50: {map50:.4f}")
        if map50_95 is not None:
            parts.append(f"mAP@50:95: {map50_95:.4f}")
        parts.append(f"time: {format_duration(epoch_time)}")
        self._clear_line()
        print("  ".join(parts), file=self.stream)

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._subscription is not None and self._events is not None:
            self._events.unsubscribe(self._subscription)
        total_seconds = time.perf_counter() - self._start
        self._clear_line()
        lines = [
            "=============== Training Finished ===============",
            f"Finished Time       : {datetime.now().strftime('%m-%d_%H-%M')}",
        ]
        if self._best_map is not None:
            lines.append(f"Best mAP@50         : {self._best_map:.4f}")
            lines.append(f"Best Epoch          : {self._best_epoch}")
        lines.append(
            f"Total Training Time : {total_seconds:.2f} sec ({total_seconds / 60:.2f} min)"
        )
        lines.append("=================================================")
        print("\n".join(lines), file=self.stream)
