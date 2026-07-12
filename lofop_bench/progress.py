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
import time
from typing import TextIO


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
    """Context manager: live per-epoch bar driven by LOFOP training events.

    Subscribes to the framework event bus' ``train.epoch_end`` topic, so it
    reflects real training progress without the harness knowing the trainer's
    internals. Safe if events never fire (the bar just stays at 0).
    """

    def __init__(self, total_epochs: int, *, prefix: str = "  training") -> None:
        self.bar = ProgressBar(total_epochs, prefix=prefix)
        self._subscription = None
        self._events = None

    def __enter__(self) -> TrainingProgress:
        from lofop.registries import EVENTS

        self._events = EVENTS
        self._subscription = EVENTS.subscribe("train.epoch_end", self._on_epoch_end)
        self.bar.update(0, "starting")
        return self

    def _on_epoch_end(self, event) -> None:
        epoch = int(event.get("epoch", 0)) + 1
        loss = event.get("loss")
        self.bar.update(epoch, f"loss={loss:.4f}" if loss is not None else "")

    def __exit__(self, *exc_info) -> None:
        self.bar.finish("training complete")
        if self._subscription is not None and self._events is not None:
            self._events.unsubscribe(self._subscription)
