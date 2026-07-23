from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from carm_roofline.core import DataType


class GPUVendor(Enum):
    """GPU vendor enumeration."""

    NVIDIA = "nvidia"
    AMD = "amd"


@dataclass(frozen=True)
class GPULaunchConfig:
    """Configuration for GPU kernel launch.

    Replaces ``BenchmarkParams.thread_affinity`` for GPU benchmarks.
    ``num_threads = blocks * threads_per_block``.
    """

    blocks: int
    threads_per_block: int = 1024
    sm_targets: int | None = None

    @property
    def num_threads(self) -> int:
        """Total number of threads = blocks x threads_per_block."""
        return self.blocks * self.threads_per_block


@dataclass(frozen=True)
class TensorPrecision:
    """Describes a tensor/matrix core precision configuration.

    ``precision_triple`` maps the A, B, and C operands to ``DataType``
    members (type-safe; no stringly-typed type names).
    ``flops_per_mma = 2 * M * N * K`` (one multiply + one add per element).
    """

    name: str
    precision_triple: tuple[DataType, DataType, DataType]  # (A_type, B_type, C_type)
    tile_mnk: tuple[int, int, int]  # (M, N, K)
    flops_per_mma: int
