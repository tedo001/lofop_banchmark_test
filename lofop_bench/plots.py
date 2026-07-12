"""Optional charts for a benchmark run (requires matplotlib).

Kept separate and import-guarded so the core harness has no plotting dependency:
``pip install matplotlib`` (or ``pip install -r requirements-plots.txt``) to
enable. Renders PNGs under ``results/``: model size vs. throughput, per-variant
latency by batch size, and a single ``summary.png`` report image that gathers
the whole run into one picture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lofop_bench.accuracy import AccuracyResult
from lofop_bench.environment import Environment
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


def plot_accuracy_curve(history: list[dict], output_dir: str | Path) -> Path | None:
    """Plot mAP@50 and training loss against epoch -> ``accuracy_curve.png``.

    This is the "should I train longer?" diagnostic: if mAP is still rising at
    the last epoch, more epochs will help; if it has flattened, the model has
    plateaued and you need a bigger variant or more data.
    """
    points = [h for h in history if h.get("map50") is not None]
    if not points:
        return None
    plt = _require_matplotlib()
    epochs = [h["epoch"] for h in points]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h["map50"] for h in points], marker="o", color="#3fb950", label="mAP@50")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP@50", color="#3fb950")
    ax.grid(True, alpha=0.3)
    losses = [h.get("loss") for h in points]
    if any(v is not None for v in losses):
        ax2 = ax.twinx()
        ax2.plot(epochs, losses, color="#58a6ff", alpha=0.7, label="train loss")
        ax2.set_ylabel("train loss", color="#58a6ff")
    ax.set_title("Accuracy vs. epoch")
    path = Path(output_dir) / "accuracy_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_summary_image(
    output_dir: str | Path,
    *,
    environment: Environment | None = None,
    structural: list[ModelReport] | None = None,
    latency: list[LatencyResult] | None = None,
    accuracy: AccuracyResult | None = None,
) -> Path:
    """Render the whole run as a single ``summary.png`` report card.

    A dark-themed dashboard: a header with the machine/version, a bar chart of
    model size, a bar chart of forward speed, and a text panel with the accuracy
    result. Panels for benchmarks that were skipped are simply left out.
    """
    plt = _require_matplotlib()
    fig = plt.figure(figsize=(11, 6.5), facecolor="#0d1117")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42, wspace=0.22)

    device = "cpu"
    if environment is not None:
        device = environment.device_name if environment.cuda_available else "CPU"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    version = environment.lofop_version if environment else "?"
    fig.suptitle("LOFOP Benchmark Report", color="#e6edf3", fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.915, f"lofop {version}   |   {device}   |   {stamp}",
             color="#8b949e", fontsize=11, ha="center")

    _bar_panel(fig.add_subplot(grid[0, 0]), structural, "parameters", 1e6,
               "Parameters (M)", "#58a6ff")
    _speed_panel(fig.add_subplot(grid[0, 1]), structural)
    _latency_panel(fig.add_subplot(grid[1, 0]), latency)
    _accuracy_panel(fig.add_subplot(grid[1, 1]), accuracy)

    path = Path(output_dir) / "summary.png"
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _style_axes(ax, title: str) -> None:
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="#e6edf3", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15, color="#8b949e")


def _short(reports: list[ModelReport]) -> list[str]:
    return [r.name.replace("lofop-detect-", "") for r in reports]


def _bar_panel(ax, reports, attr, scale, title, color) -> None:
    _style_axes(ax, title)
    if not reports:
        ax.text(0.5, 0.5, "skipped", color="#8b949e", ha="center", va="center")
        return
    names = _short(reports)
    values = [getattr(r, attr) / scale for r in reports]
    ax.bar(names, values, color=color)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.1f}", color="#e6edf3", ha="center", va="bottom", fontsize=8)


def _speed_panel(ax, reports) -> None:
    _style_axes(ax, "Forward speed (FPS)")
    if not reports:
        ax.text(0.5, 0.5, "skipped", color="#8b949e", ha="center", va="center")
        return
    names = _short(reports)
    values = [(r.gpu_fps or r.cpu_fps or 0.0) for r in reports]
    ax.bar(names, values, color="#3fb950")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.0f}", color="#e6edf3", ha="center", va="bottom", fontsize=8)


def _latency_panel(ax, results) -> None:
    _style_axes(ax, "Latency p50 per image (ms)")
    if not results:
        ax.text(0.5, 0.5, "skipped", color="#8b949e", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for result in results:
        batches = sorted(result.batch_latencies_ms)
        p50 = [result.batch_latencies_ms[b]["p50"] for b in batches]
        ax.plot(batches, p50, marker="o", label=result.variant)
    ax.set_xlabel("batch size", color="#8b949e", fontsize=8)
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")


def _accuracy_panel(ax, accuracy) -> None:
    _style_axes(ax, "Accuracy")
    ax.set_xticks([])
    ax.set_yticks([])
    if accuracy is None:
        ax.text(0.5, 0.5, "skipped", color="#8b949e", ha="center", va="center")
        return
    lines = [
        f"variant   lofop-detect-{accuracy.variant}",
        f"dataset   {accuracy.dataset}",
        f"epochs    {accuracy.epochs} @ {accuracy.image_size}px ({accuracy.device})",
        "",
        f"mAP@50      {accuracy.map50:.4f}",
        f"mAP@50:95   {accuracy.map50_95:.4f}",
        f"precision   {accuracy.precision:.4f}",
        f"recall      {accuracy.recall:.4f}",
        f"F1          {accuracy.f1:.4f}",
    ]
    ax.text(0.04, 0.95, "\n".join(lines), color="#e6edf3", fontsize=10, va="top",
            family="monospace", transform=ax.transAxes)
