"""Structural benchmarks: size and speed of every LOFOP-Detect variant.

Uses LOFOP's public measurement instrument (``lofop.utils.benchmark_model`` +
``render_table``), then writes markdown/CSV/JSON itself so the harness depends
only on the stable public API of the *published* ``lofop`` package -- not on any
particular version's internal export helpers. Accuracy columns stay empty:
structural metrics never depend on trained weights, and this harness never
invents accuracy numbers (see ``accuracy.py`` for measured ones).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import lofop.models  # noqa: F401  (registers model components)
from lofop.core.config import Config
from lofop.registries import HUB
from lofop.utils import ModelReport, benchmark_model, render_table

# The variant configs ship inside the installed lofop package.
_CONFIG_DIR = Path(lofop.models.__file__).resolve().parent.parent / "configs" / "lofop-detect"

_FIELDS = (
    "name", "parameters", "flops", "flops_g", "size_mb", "cpu_fps", "gpu_fps",
    "map50", "map50_95", "precision", "recall", "f1",
)


def variant_configs() -> list[Path]:
    """The shippable model variants (every config except the shared base)."""
    return [p for p in sorted(_CONFIG_DIR.glob("*.yaml")) if p.stem != "base"]


def benchmark_variant(config_path: Path, *, image_size: int) -> ModelReport:
    model = HUB.build(Config.load(config_path).model)
    return benchmark_model(model, f"lofop-detect-{config_path.stem}", image_size=image_size)


def _record(report: ModelReport) -> dict[str, object]:
    """Flatten a ModelReport to a CSV/JSON row using only guaranteed fields."""
    acc = report.accuracy
    return {
        "name": report.name,
        "parameters": report.parameters,
        "flops": report.flops,
        "flops_g": round(report.flops / 1e9, 4),
        "size_mb": round(report.size_mb, 4),
        "cpu_fps": round(report.cpu_fps, 3) if report.cpu_fps else None,
        "gpu_fps": round(report.gpu_fps, 3) if report.gpu_fps else None,
        "map50": round(acc.map50, 4) if acc else None,
        "map50_95": round(acc.map50_95, 4) if acc else None,
        "precision": round(acc.precision, 4) if acc else None,
        "recall": round(acc.recall, 4) if acc else None,
        "f1": round(getattr(acc, "f1", 0.0), 4) if acc else None,
    }


def _write(reports: list[ModelReport], output_dir: Path, basename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.md").write_text(render_table(reports), encoding="utf-8")

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(_FIELDS))
    writer.writeheader()
    for report in reports:
        writer.writerow(_record(report))
    (output_dir / f"{basename}.csv").write_text(buffer.getvalue(), encoding="utf-8")

    (output_dir / f"{basename}.json").write_text(
        json.dumps([_record(r) for r in reports], indent=2) + "\n", encoding="utf-8"
    )


def run_structural(
    output_dir: str | Path, *, image_size: int = 640, basename: str = "structural",
) -> list[ModelReport]:
    """Benchmark every variant and write ``structural.{md,csv,json}``."""
    reports = [benchmark_variant(path, image_size=image_size) for path in variant_configs()]
    _write(reports, Path(output_dir), basename)
    return reports


if __name__ == "__main__":
    results = run_structural(Path(__file__).resolve().parent.parent / "results")
    print(render_table(results))
