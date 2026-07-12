"""Latency and throughput benchmark, GPU-aware.

Measures what a deployment actually cares about: per-image latency percentiles
(p50/p90/p99), throughput at several batch sizes, and -- on CUDA -- peak VRAM.
Uses CUDA events for correct GPU timing (wall-clock time is wrong for async GPU
work) and an optional autocast path so you can compare FP32 vs mixed precision
on an RTX card.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

import lofop.models  # noqa: F401  (registers model components)
from lofop.core.config import Config
from lofop.registries import HUB

from lofop_bench.progress import ProgressBar

_CONFIG_DIR = Path(lofop.models.__file__).resolve().parent.parent / "configs" / "lofop-detect"


@dataclass
class LatencyResult:
    variant: str
    device: str
    image_size: int
    amp: bool
    batch_latencies_ms: dict[int, dict[str, float]] = field(default_factory=dict)
    throughput_fps: dict[int, float] = field(default_factory=dict)
    peak_vram_mb: float | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_once(model: nn.Module, images: torch.Tensor, device: torch.device, amp: bool) -> float:
    """Milliseconds for one forward pass, using CUDA events on GPU."""
    with torch.autocast(device.type, enabled=amp), torch.inference_mode():
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(images)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        begin = time.perf_counter()
        model(images)
        return (time.perf_counter() - begin) * 1e3


def benchmark_latency(
    config_path: Path,
    *,
    device: str,
    image_size: int = 640,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    warmup: int = 5,
    iters: int = 30,
    amp: bool = False,
) -> LatencyResult:
    dev = torch.device(device)
    model = HUB.build(Config.load(config_path).model).eval().to(dev)
    result = LatencyResult(
        variant=config_path.stem, device=dev.type, image_size=image_size, amp=amp,
    )
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for batch in batch_sizes:
        images = torch.randn(batch, 3, image_size, image_size, device=dev)
        for _ in range(warmup):
            _time_once(model, images, dev, amp)
        _sync(dev)
        times = [_time_once(model, images, dev, amp) for _ in range(iters)]
        times.sort()
        per_image = [t / batch for t in times]
        result.batch_latencies_ms[batch] = {
            "p50": round(statistics.median(per_image), 3),
            "p90": round(per_image[int(0.9 * (len(per_image) - 1))], 3),
            "p99": round(per_image[int(0.99 * (len(per_image) - 1))], 3),
            "mean": round(statistics.fmean(per_image), 3),
        }
        result.throughput_fps[batch] = round(batch * 1000.0 / statistics.median(times), 1)

    if dev.type == "cuda":
        result.peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
    return result


def variant_configs() -> list[Path]:
    return [p for p in sorted(_CONFIG_DIR.glob("*.yaml")) if p.stem != "base"]


def run_latency(
    output_dir: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    batch_sizes: tuple[int, ...] = (1, 4, 8),
    amp: bool = False,
    basename: str = "latency",
) -> list[LatencyResult]:
    """Benchmark latency for every variant and write ``latency.{md,json}``."""
    variants = variant_configs()
    bar = ProgressBar(len(variants), prefix=f"  latency ({device})")
    results = []
    for index, path in enumerate(variants):
        bar.update(index, f"timing {path.stem}")
        results.append(
            benchmark_latency(
                path, device=device, image_size=image_size,
                batch_sizes=batch_sizes, amp=amp,
            )
        )
    bar.finish()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / f"{basename}.md").write_text(render_latency_markdown(results), encoding="utf-8")
    return results


def render_latency_markdown(results: list[LatencyResult]) -> str:
    if not results:
        return "# LOFOP latency benchmark\n\n(no results)\n"
    device = results[0].device
    amp = results[0].amp
    lines = [
        "# LOFOP latency benchmark",
        "",
        f"Device `{device}`, {results[0].image_size}px, AMP={amp}. "
        "Per-image latency in ms (lower is better); throughput in FPS.",
        "",
        "| Variant | Batch | p50 (ms) | p90 (ms) | p99 (ms) | FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for batch, stats in result.batch_latencies_ms.items():
            lines.append(
                f"| lofop-detect-{result.variant} | {batch} | {stats['p50']} | "
                f"{stats['p90']} | {stats['p99']} | {result.throughput_fps[batch]} |"
            )
    if results[0].peak_vram_mb is not None:
        vram = ", ".join(f"{r.variant}: {r.peak_vram_mb} MB" for r in results)
        lines += ["", f"Peak VRAM (all batch sizes): {vram}."]
    return "\n".join(lines) + "\n"
