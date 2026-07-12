"""Structural benchmarks: size and speed of every LOFOP-Detect variant.

Reuses LOFOP's own measurement instrument (``lofop.utils.benchmark``) so the
numbers here match what ``lofop benchmark`` reports. Accuracy columns stay empty
-- structural metrics never depend on trained weights, and this harness never
invents accuracy numbers (see ``accuracy.py`` for measured ones).
"""

from __future__ import annotations

from pathlib import Path

import lofop.models  # noqa: F401  (registers model components)
from lofop.core.config import Config
from lofop.registries import HUB
from lofop.utils import ModelReport, benchmark_model, render_table, write_reports

# The variant configs ship inside the installed lofop package.
_CONFIG_DIR = Path(lofop.models.__file__).resolve().parent.parent / "configs" / "lofop-detect"


def variant_configs() -> list[Path]:
    """The shippable model variants (every config except the shared base)."""
    return [p for p in sorted(_CONFIG_DIR.glob("*.yaml")) if p.stem != "base"]


def benchmark_variant(config_path: Path, *, image_size: int) -> ModelReport:
    model = HUB.build(Config.load(config_path).model)
    return benchmark_model(model, f"lofop-detect-{config_path.stem}", image_size=image_size)


def run_structural(
    output_dir: str | Path, *, image_size: int = 640, basename: str = "structural",
) -> list[ModelReport]:
    """Benchmark every variant and write ``structural.{md,csv,json}``."""
    reports = [benchmark_variant(path, image_size=image_size) for path in variant_configs()]
    write_reports(reports, output_dir, basename=basename)
    return reports


if __name__ == "__main__":
    results = run_structural(Path(__file__).resolve().parent.parent / "results")
    print(render_table(results))
