"""Capture the hardware/software environment a benchmark ran on.

Benchmark numbers are meaningless without the machine that produced them, so
every run records the CPU/GPU, CUDA/torch versions, and LOFOP version alongside
the results. This makes a committed ``results/`` set reproducible and lets you
tell an RTX 4060 run apart from a CI CPU run at a glance.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    lofop_version: str
    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str
    cpu_threads: int

    def to_markdown(self) -> str:
        device = self.device_name if self.cuda_available else f"CPU ({self.cpu_threads} threads)"
        return (
            "# Benchmark environment\n\n"
            f"- LOFOP: {self.lofop_version}\n"
            f"- Python: {self.python_version}\n"
            f"- Platform: {self.platform}\n"
            f"- PyTorch: {self.torch_version}\n"
            f"- CUDA: {self.cuda_version or 'n/a'}\n"
            f"- Device: {device}\n"
        )


def capture() -> Environment:
    import torch

    import lofop

    cuda = torch.cuda.is_available()
    return Environment(
        lofop_version=lofop.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        cuda_version=torch.version.cuda if cuda else None,
        device_name=torch.cuda.get_device_name(0) if cuda else "cpu",
        cpu_threads=torch.get_num_threads(),
    )


def write_environment(output_dir: str | Path, *, basename: str = "environment") -> Environment:
    """Capture the environment and write ``environment.{md,json}``."""
    env = capture()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{basename}.md").write_text(env.to_markdown(), encoding="utf-8")
    (output_dir / f"{basename}.json").write_text(
        json.dumps(asdict(env), indent=2) + "\n", encoding="utf-8"
    )
    return env
