"""Mixed benchmark suite for arithmetic intensity sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..benchmark import MixedBenchmark
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from context import CARMContext


@dataclass
class MixedBenchmarkSuite(ISABenchmarkSuite):
    """Suite for mixed arithmetic+memory benchmarks (TestType.MIXED).

    Sweeps arithmetic intensity to characterize performance across
    the bandwidth-limited to compute-limited spectrum.

    TODO: Implement when MixedBenchmarkParams is defined.
    """

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> MixedBenchmarkSuite:
        """Generate mixed microbenchmarks for a single ISA.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for.

        Returns:
            MixedBenchmarkSuite with AI-sweep benchmarks.

        Raises:
            NotImplementedError: Mixed benchmarks not yet implemented.
        """
        raise NotImplementedError(
            "Mixed benchmark generation not yet implemented. "
            "Requires MixedBenchmarkParams definition and ISA.generate_mixed() method."
        )

    def get_benchmarks_by_intensity(self) -> dict[float, MixedBenchmark]:
        """Group mixed benchmarks by arithmetic intensity.

        Returns:
            Dictionary mapping AI value to benchmark.
        """
        # TODO: Implement when mixed benchmarks are defined
        return {}
