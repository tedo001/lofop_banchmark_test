"""Simple, dependency-free progress bars for benchmark runs.

Prints a carriage-return-updating bar to stderr (so it never pollutes the
markdown/JSON that goes to stdout). Two helpers:

* :class:`ProgressBar` -- a generic ``current/total`` bar with elapsed/ETA.
* :class:`TrainingProgress` -- a context manager that shows a live per-epoch
  bar during ``Detector.train`` by listening to LOFOP's ``train.epoch_end``
  event, so you get feedback through the long training step without touching
  the framework.
"""

from __future__ import annotations

import sys
import threading
import time
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
    """Context manager: per-epoch training log driven by LOFOP events.

    Subscribes to the framework event bus' ``train.epoch_end`` topic, so it
    reflects real training progress without the harness knowing the trainer's
    internals. Prints one line per finished epoch (loss, mAP, epoch time,
    elapsed, ETA). Between epochs a background thread animates a live bar a few
    times a second so a long epoch never looks frozen: a bouncing bar during the
    first epoch (unknown length) and an estimated-percent fill afterwards (once
    the previous epoch's duration is known).
    """

    _SPINNER = "|/-\\"
    _BAR_WIDTH = 22
    _BLOCK = 4

    def __init__(
        self, total_epochs: int, *, prefix: str = "training", interval: float = 0.2,
        stream: TextIO | None = None,
    ) -> None:
        self.total = max(int(total_epochs), 1)
        self.prefix = prefix
        self.interval = interval
        self.stream = stream or sys.stderr
        self._subscription = None
        self._events = None
        self._start = 0.0
        self._last_epoch_time = 0.0
        self._est_epoch = 0.0  # duration of the last finished epoch, for the estimate
        self._completed = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> TrainingProgress:
        from lofop.registries import EVENTS

        self._events = EVENTS
        self._start = self._last_epoch_time = time.perf_counter()
        self._subscription = EVENTS.subscribe("train.epoch_end", self._on_epoch_end)
        print(
            f"{self.prefix}: {self.total} epochs (one line per epoch; live bar below)",
            file=self.stream,
        )
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _indeterminate_bar(self, tick: int) -> str:
        """A block bouncing left<->right, so it visibly moves without a known %."""
        span = self._BAR_WIDTH - self._BLOCK
        cycle = tick % (2 * span or 1)
        pos = cycle if cycle <= span else 2 * span - cycle
        return "-" * pos + "#" * self._BLOCK + "-" * (span - pos)

    def _estimated_bar(self, fraction: float) -> str:
        filled = int(min(fraction, 1.0) * self._BAR_WIDTH)
        return "#" * filled + "-" * (self._BAR_WIDTH - filled)

    def _beat(self) -> None:
        tick = 0
        while not self._stop.wait(self.interval):
            tick += 1
            spin = self._SPINNER[tick % len(self._SPINNER)]
            in_epoch = time.perf_counter() - self._last_epoch_time
            if self._est_epoch > 0:  # we know how long an epoch takes -> estimate %
                fraction = in_epoch / self._est_epoch
                body = f"[{self._estimated_bar(fraction)}] ~{min(fraction, 0.999) * 100:4.1f}%"
            else:  # first epoch: unknown length -> animated bounce
                body = f"[{self._indeterminate_bar(tick)}]"
            self.stream.write(
                f"\r  {spin} epoch {self._completed + 1}/{self.total}  {body}  "
                f"{format_duration(in_epoch)}   "
            )
            self.stream.flush()

    def _on_epoch_end(self, event) -> None:
        now = time.perf_counter()
        self._completed = int(event.get("epoch", 0)) + 1
        epoch_time = now - self._last_epoch_time
        self._last_epoch_time = now
        self._est_epoch = epoch_time  # use the last epoch's time to estimate the next
        elapsed = now - self._start
        eta = (elapsed / self._completed) * (self.total - self._completed)
        loss = event.get("loss")
        metrics = event.get("metrics") or {}
        parts = [f"Epoch {self._completed:>3}/{self.total}"]
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if metrics.get("map50") is not None:
            parts.append(f"mAP50 {metrics['map50']:.4f}")
        parts += [
            f"epoch {format_duration(epoch_time)}",
            f"elapsed {format_duration(elapsed)}",
            f"ETA {format_duration(eta)}",
        ]
        self.stream.write("\r" + " " * 78 + "\r")  # clear the heartbeat line
        print("  " + "   ".join(parts), file=self.stream)

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._subscription is not None and self._events is not None:
            self._events.unsubscribe(self._subscription)
        self.stream.write("\r" + " " * 78 + "\r")
        print(
            f"  {self.prefix}: done in {format_duration(time.perf_counter() - self._start)}",
            file=self.stream,
        )
