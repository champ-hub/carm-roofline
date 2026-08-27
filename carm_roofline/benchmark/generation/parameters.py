from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from carm_roofline.core import ArithmeticIntensity, ArithmeticOperation, Bytes, Operations

if TYPE_CHECKING:
    from carm_roofline.benchmark.benchmarking import LoadStoreRatio

from carm_roofline.core import DataType


class BenchParamError(Exception):
    pass


class MemoryLayoutMode(Enum):
    single = "single"
    split = "split"


@dataclass
class BenchmarkParams:
    """Base class for benchmark generation parameters.

    Encapsulates common parameters for all benchmark types.
    """

    data_type: DataType  # Data type to use in the benchmark (e.g., i8, f32)
    thread_affinity: list[int]

    @property
    def num_threads(self) -> int:
        return len(self.thread_affinity)

    def __post_init__(self) -> None:
        if self.num_threads < 1:
            raise BenchParamError("Number of threads must be at least 1")


@dataclass
class ArithmeticBenchmarkParams(BenchmarkParams):
    """Parameters for arithmetic benchmark generation.

    Specifies the arithmetic operation and number of operations to perform.

    Example:
        params = ArithmeticBenchmarkParams(
            data_type=DataType.f32,
            operation=ArithmeticOperation.fma,
            num_ops=Operations(1000),
            thread_affinity=[0, 1, 2, 3],
        )
        spec = isa.generate_arithmetic(params)
    """

    operation: ArithmeticOperation
    num_ops: Operations

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.num_ops <= Operations(0):
            raise BenchParamError("At least one operation (num_ops) is required")


@dataclass
class MemoryAccessParams(BenchmarkParams):
    """Common parameters for benchmarks that access memory."""

    load_store_ratio: LoadStoreRatio
    size_per_thread: Bytes
    memory_level_name: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.size_per_thread <= Bytes(0):
            raise BenchParamError("Size of the working set must be positive")
        self.num_ld = self.load_store_ratio.loads
        self.num_st = self.load_store_ratio.stores
        if self.num_ld < 0 or self.num_st < 0 or self.num_ld + self.num_st < 1:
            raise BenchParamError("Number of loads and stores must be positive and num_ld + num_st > 0")


@dataclass
class MemoryBenchmarkParams(MemoryAccessParams):
    """Parameters for memory benchmark generation."""

    layout_mode: MemoryLayoutMode = MemoryLayoutMode.split


@dataclass
class MixedBenchmarkParams(MemoryAccessParams):
    """Parameters for mixed arithmetic and memory benchmark generation."""

    operation: ArithmeticOperation
    point_index: int
    requested_arithmetic_intensity: ArithmeticIntensity
    num_arithmetic_instructions: int
    memory_pattern_repeats: int
    achieved_arithmetic_intensity: ArithmeticIntensity
    layout_mode: MemoryLayoutMode = MemoryLayoutMode.split

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.point_index < 0:
            raise BenchParamError("Mixed benchmark point index must be nonnegative")
        if self.num_arithmetic_instructions < 1 or self.memory_pattern_repeats < 1:
            raise BenchParamError("Mixed benchmark instruction counts must be positive")
        if self.requested_arithmetic_intensity <= ArithmeticIntensity(0):
            raise BenchParamError("Requested arithmetic intensity must be positive")
        if self.achieved_arithmetic_intensity <= ArithmeticIntensity(0):
            raise BenchParamError("Achieved arithmetic intensity must be positive")
