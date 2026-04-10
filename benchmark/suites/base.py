"""Abstract base class for ISA benchmark suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from units import Bandwidth, Performance

if TYPE_CHECKING:
    from context import CARMContext

    from ..benchmark import ArithmeticBenchmark, BaseBenchmark, MemoryBenchmark, MixedBenchmark


@dataclass
class ISABenchmarkSuite(ABC):
    """Groups benchmarks by ISA for easier result processing and roofline analysis.

    Each ISA (e.g., avx2, avx512, neon) gets its own suite containing all
    benchmarks (arithmetic, memory, mixed) for that ISA.

    This is an abstract base class - use specialized subclasses:
    - ArithmeticBenchmarkSuite for TestType.ARITHMETIC
    - MemoryBenchmarkSuite for TestType.MEMORY
    - RooflineBenchmarkSuite for TestType.ROOFLINE
    - MixedBenchmarkSuite for TestType.MIXED
    """

    isa_name: str
    benchmarks: dict[str, BaseBenchmark] = field(default_factory=dict)

    @classmethod
    @abstractmethod
    def generate(cls, context: CARMContext, isa_name: str) -> ISABenchmarkSuite:
        """Generate benchmarks for this suite type.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for.

        Returns:
            ISABenchmarkSuite subclass instance with generated benchmarks.
        """

    def get_peak_performance(self) -> Performance:
        """Get peak GOPS from arithmetic benchmarks."""

        arith_benchmarks = self.get_arithmetic_benchmarks()
        max_perf = Performance(0.0)

        if not arith_benchmarks:
            raise ValueError("No arithmetic benchmarks found for peak performance calculation")

        for name, bench in arith_benchmarks.items():
            if bench.results is None:
                raise ValueError(f"Arithmetic benchmark '{name}' missing performance result")

            max_perf = max(max_perf, bench.results.performance)

        return max_perf

    def get_bandwidth_by_level(self) -> dict[str, Bandwidth]:
        """Get peak bandwidth by cache level (L1/L2/L3/DRAM)."""

        result: dict[str, Bandwidth] = {}

        # Group by cache level
        for bench in self.get_memory_benchmarks().values():
            if bench.cache_level is None:
                raise ValueError(
                    f"Memory benchmark {bench.name} missing cache level annotation for roofline benchmarking"
                )
            if bench.results is None:
                raise ValueError(f"Memory benchmark {bench.name} missing bandwidth result for roofline benchmarking")
            result[bench.cache_level] = bench.results.bandwidth

        return result

    def add_benchmark(self, name: str, benchmark: BaseBenchmark) -> None:
        """Add a benchmark to this ISA's suite."""
        self.benchmarks[name] = benchmark

    def get_arithmetic_benchmarks(self) -> dict[str, ArithmeticBenchmark]:
        """Return only arithmetic benchmarks from this suite."""
        from ..benchmark import ArithmeticBenchmark

        return {name: bench for name, bench in self.benchmarks.items() if isinstance(bench, ArithmeticBenchmark)}

    def get_memory_benchmarks(self) -> dict[str, MemoryBenchmark]:
        """Return only memory benchmarks from this suite."""
        from ..benchmark import MemoryBenchmark

        return {name: bench for name, bench in self.benchmarks.items() if isinstance(bench, MemoryBenchmark)}

    def get_mixed_benchmarks(self) -> dict[str, MixedBenchmark]:
        """Return only mixed benchmarks from this suite."""
        from ..benchmark import MixedBenchmark

        return {name: bench for name, bench in self.benchmarks.items() if isinstance(bench, MixedBenchmark)}

    def all_results_populated(self) -> bool:
        """Check if all benchmarks in the suite have results."""
        return all(bench.results is not None for bench in self.benchmarks.values())

    def merge(self, other: ISABenchmarkSuite) -> None:
        """Merge another suite's benchmarks into this one.

        Useful for combining arithmetic, memory, and mixed benchmarks for the same ISA.

        Args:
            other: Another ISABenchmarkSuite for the same ISA

        Raises:
            ValueError: If the other suite is for a different ISA
        """
        if other.isa_name != self.isa_name:
            raise ValueError(f"Cannot merge suite for ISA '{other.isa_name}' into suite for ISA '{self.isa_name}'")
        # Add all benchmarks from other suite
        for name, benchmark in other.benchmarks.items():
            if name in self.benchmarks:
                raise ValueError(f"Benchmark '{name}' already exists in suite for ISA '{self.isa_name}'")
            self.benchmarks[name] = benchmark

    @staticmethod
    def merge_suites(*suite_dicts: dict[str, ISABenchmarkSuite]) -> dict[str, ISABenchmarkSuite]:
        """Merge multiple ISA benchmark suite dictionaries by ISA name.

        Combines suites from different benchmark types (arithmetic, memory, mixed)
        into unified suites per ISA. Useful for roofline analysis which needs
        both arithmetic and memory benchmarks for the same ISA.

        DEPRECATED: Prefer using RooflineBenchmarkSuite.generate() which handles
        composition automatically. This method is kept for backward compatibility.

        Args:
            *suite_dicts: Multiple dictionaries mapping ISA names to ISABenchmarkSuite objects

        Returns:
            A single dictionary with merged suites, keyed by ISA name.
            The returned suites will be instances of the first suite type encountered
            for each ISA.

        Example:
            >>> arith_suites = {isa: ArithmeticBenchmarkSuite.generate(ctx, isa) for isa in isas}
            >>> mem_suites = {isa: MemoryBenchmarkSuite.generate(ctx, isa) for isa in isas}
            >>> merged = ISABenchmarkSuite.merge_suites(arith_suites, mem_suites)
            >>> for isa_name, suite in merged.items():
            ...     arith = suite.get_arithmetic_benchmarks()
            ...     mem = suite.get_memory_benchmarks()
            ...     # Process both for roofline...

        Raises:
            ValueError: If benchmark names conflict across suites for the same ISA
        """
        merged: dict[str, ISABenchmarkSuite] = {}

        for suite_dict in suite_dicts:
            for isa_name, suite in suite_dict.items():
                if isa_name not in merged:
                    # Use the same type as the incoming suite
                    merged[isa_name] = type(suite)(isa_name=isa_name)
                # Merge benchmarks from this suite
                merged[isa_name].merge(suite)

        return merged
