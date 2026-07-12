"""Optional charts for a benchmark run (requires matplotlib).

Kept separate and import-guarded so the core harness has no plotting dependency:
``pip install matplotlib`` (or ``pip install -r requirements-plots.txt``) to
enable. Renders two PNGs under ``results/``: model size vs. throughput, and
per-variant latency by batch size.
"""

from __future__ import annotations

from pathlib import Path

from lofop_bench.latency import LatencyResult
from lofop.utils import ModelReport


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed on a CI box or laptop
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plots; install it with "
            "`pip install matplotlib` or `pip install -r requirements-plots.txt`"
        ) from exc


def plot_size_vs_speed(reports: list[ModelReport], output_dir: str | Path) -> Path:
    """Scatter of parameter count vs. CPU/GPU FPS for each variant."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    for report in reports:
        fps = report.gpu_fps or report.cpu_fps or 0.0
        ax.scatter(report.parameters / 1e6, fps)
        ax.annotate(report.name, (report.parameters / 1e6, fps),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Forward FPS")
    ax.set_title("LOFOP-Detect: size vs. speed")
    ax.grid(True, alpha=0.3)
    path = Path(output_dir) / "size_vs_speed.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_latency_by_batch(results: list[LatencyResult], output_dir: str | Path) -> Path:
    """Grouped bars of p50 per-image latency by batch size, per variant."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 4))
    for result in results:
        batches = sorted(result.batch_latencies_ms)
        p50 = [result.batch_latencies_ms[b]["p50"] for b in batches]
        ax.plot(batches, p50, marker="o", label=f"lofop-detect-{result.variant}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("p50 latency per image (ms)")
    ax.set_title(f"Latency by batch size ({results[0].device})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = Path(output_dir) / "latency_by_batch.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
