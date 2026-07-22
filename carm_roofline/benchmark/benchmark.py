"""
Benchmark code, metadata and result data structures. For benchmark configuration and argument parsing, see
benchmark/benchmarking.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from carm_roofline.core import ArithmeticIntensity, Bandwidth, Bytes, Performance, Seconds
from carm_roofline.output_utils import debug
from carm_roofline.test_bench.builder import MicrobenchmarkFunctionSpec

from .generation import ArithmeticBenchmarkParams, BenchmarkParams, MemoryBenchmarkParams

# --- Result Classes ---


@dataclass
class BaseBenchmarkResult(ABC):
    """Base class for benchmark execution results."""

    time_taken: Seconds  # Total time in milliseconds
    num_repetitions: int  # How many times the test repeated


@dataclass
class ArithmeticBenchmarkResult(BaseBenchmarkResult):
    """Results specific to arithmetic benchmarks."""

    performance: Performance


@dataclass
class MemoryBenchmarkResult(BaseBenchmarkResult):
    """Results specific to memory benchmarks."""

    bandwidth: Bandwidth  # Memory bandwidth in GB/s
    cache_level: str | None = None  # Which cache level was tested (L1/L2/L3/DRAM)


@dataclass
class MixedBenchmarkResult(BaseBenchmarkResult):
    """Results for mixed arithmetic+memory benchmarks."""

    performance: Performance
    arithmetic_intensity: ArithmeticIntensity


# --- Benchmark Classes ---


@dataclass
class BaseBenchmark(ABC):
    """Base class for all benchmark types.

    Provides common interface for benchmark execution and result processing.
    """

    params: BenchmarkParams
    spec: MicrobenchmarkFunctionSpec
    results: BaseBenchmarkResult | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        """The authoritative benchmark name (from the C function name).

        This is the single source of truth for benchmark identity.
        It's guaranteed to match what's compiled in the binary.
        """
        return self.spec.function_name

    @abstractmethod
    def process_results(self, time_taken: Seconds, num_repetitions: int) -> None:
        """Populate the benchmark results from execution data.

        Args:
            time_taken: Execution time in seconds
            num_repetitions: Number of times the benchmark loop repeated
        """


@dataclass
class ArithmeticBenchmark(BaseBenchmark):
    """Arithmetic benchmark with arithmetic operations."""

    params: ArithmeticBenchmarkParams
    results: ArithmeticBenchmarkResult | None = field(default=None, init=False)

    def process_results(self, time_taken: Seconds, num_repetitions: int) -> None:
        """Calculate aggregate GOPS from execution results.

        Note: num_repetitions is per-thread; multiply by num_threads for total work.
        This gives aggregate throughput across all threads (standard for roofline analysis).
        """
        operations = self.params.num_ops * num_repetitions * self.params.num_threads
        performance = Performance.from_ops_per_second(operations, time_taken)
        self.results = ArithmeticBenchmarkResult(
            time_taken=time_taken,
            num_repetitions=num_repetitions,
            performance=performance,
        )
        debug(
            f"Processed arithmetic benchmark '{self.name}': {time_taken}, {num_repetitions} reps, "
            f"{self.params.num_threads} threads, {performance}"
        )


@dataclass
class MemoryBenchmark(BaseBenchmark):
    """Memory benchmark with load/store operations."""

    params: MemoryBenchmarkParams
    working_set_bytes: Bytes
    "Working set size across all threads, used for bandwidth calculation"
    results: MemoryBenchmarkResult | None = field(default=None, init=False)
    cache_level: str | None = None  # L1, ..., DRAM, etc.

    @property
    def name(self) -> str:
        """The authoritative benchmark name (from the C function name).

        This is the single source of truth for benchmark identity.
        It's guaranteed to match what's compiled in the binary.
        For memory benchmarks, the function_name is already enriched with cache level and size.
        """
        return self.spec.function_name

    def process_results(self, time_taken: Seconds, num_repetitions: int) -> None:
        """Calculate aggregate memory bandwidth from execution results.

        Note: Computation accounts for both code-generation and runtime scaling:
        - params.repeats: Static repetitions in generated assembly
        - num_repetitions: Runtime calibrated repetitions (per-thread)
        - num_threads: Thread count for aggregate throughput
        Total bytes = (loads + stores) * repeats * num_repetitions * num_threads * bytes_per_element
        """
        total_memory_transacted = self.working_set_bytes * num_repetitions
        bandwidth = Bandwidth.from_bytes_per_second(total_memory_transacted, time_taken)
        self.results = MemoryBenchmarkResult(
            time_taken=time_taken,
            num_repetitions=num_repetitions,
            bandwidth=bandwidth,
            cache_level=self.cache_level,
        )
        debug(
            f"Processed memory benchmark '{self.name}': {time_taken}, {num_repetitions} reps, "
            f"{self.params.num_threads} threads, {bandwidth}"
        )


@dataclass
class MixedBenchmark(BaseBenchmark):
    """Mixed arithmetic+memory benchmark for roofline analysis."""

    params: BenchmarkParams  # TODO: Define MixedBenchmarkParams
    results: MixedBenchmarkResult | None = field(default=None, init=False)

    def process_results(self, time_taken: Seconds, num_repetitions: int) -> None:
        """Calculate GOPS from execution results."""
        # TODO: Implement when MixedBenchmarkParams is defined
        raise NotImplementedError("Mixed benchmark result processing not yet implemented")


# --- ISA Grouping ---
# ISABenchmarkSuite moved to benchmark/suites/base.py for better organization
# Import here for backward compatibility with "from benchmark.benchmark import ISABenchmarkSuite"
from .suites.base import ISABenchmarkSuite  # noqa: E402

__all__ = [
    "ArithmeticBenchmark",
    "ArithmeticBenchmarkResult",
    "BaseBenchmark",
    "BaseBenchmarkResult",
    "ISABenchmarkSuite",
    "MemoryBenchmark",
    "MemoryBenchmarkResult",
    "MixedBenchmark",
    "MixedBenchmarkResult",
]
