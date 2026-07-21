"""Roofline benchmark suite combining arithmetic and memory measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core import ArithmeticIntensity, Bandwidth

from .arithmetic import ArithmeticBenchmarkSuite
from .base import ISABenchmarkSuite
from .memory import MemoryBenchmarkSuite

if TYPE_CHECKING:
    from context import CARMContext


@dataclass
class RooflineBenchmarkSuite(ISABenchmarkSuite):
    """Suite for roofline model benchmarks (TestType.ROOFLINE).

    Combines arithmetic (peak FP) and memory (cache hierarchy bandwidth) benchmarks
    to construct the roofline performance model.
    """

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> RooflineBenchmarkSuite:
        """Generate roofline microbenchmarks (arithmetic + memory) for a single ISA.

        Creates both arithmetic and memory benchmarks, then merges them into
        a unified suite for roofline analysis.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for.

        Returns:
            RooflineBenchmarkSuite with both arithmetic and memory benchmarks.

        Raises:
            ValueError: Unsupported precision or unknown ISA name.
        """
        # Generate arithmetic benchmarks
        arith_suite = ArithmeticBenchmarkSuite.generate(context, isa_name)

        # Generate memory benchmarks
        mem_suite = MemoryBenchmarkSuite.generate(context, isa_name)

        # Create roofline suite and merge both
        suite = cls(isa_name=isa_name)
        suite.merge(arith_suite)
        suite.merge(mem_suite)

        return suite

    def get_bandwidth_by_level(self) -> dict[str, Bandwidth]:
        """Get peak bandwidth for each cache level from memory benchmarks.

        Returns:
            Dictionary mapping cache level to peak bandwidth in GB/s.
        """
        result: dict[str, Bandwidth] = {}

        # Group by cache level
        for bench in self.get_memory_benchmarks().values():
            if bench.cache_level is None:
                raise ValueError(
                    f"Memory benchmark {bench.name} missing cache level annotation for roofline benchmarki>ng"
                )
            if bench.results is None:
                raise ValueError(f"Memory benchmark {bench.name} missing bandwidth result for roofline benchmarking")
            result[bench.cache_level] = bench.results.bandwidth

        return result

    def compute_ridge_points(self) -> dict[str, ArithmeticIntensity]:
        """Compute ridge points where cache levels transition from bandwidth to compute bound.

        Ridge point = (AI, Performance) where AI = peak_gops / peak_bandwidth.

        Returns:
            Dictionary mapping cache level to arithmetic intensity at ridge point.
        """
        peak_performance = self.get_peak_performance()
        if peak_performance is None:
            return {}

        bandwidth_by_level = self.get_bandwidth_by_level()
        ridge_points = {}

        for level, bandwidth in bandwidth_by_level.items():
            # AI at ridge point (GOPS/s / GB/s)
            # Seconds cancel out, leaving us with OPS/byte which is the definition of arithmetic intensity
            arithmetic_intensity = ArithmeticIntensity(peak_performance.value / bandwidth.value)
            # Performance at ridge point is the peak GOPS
            ridge_points[level] = arithmetic_intensity

        return ridge_points
