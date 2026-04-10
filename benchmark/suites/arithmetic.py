"""Arithmetic benchmark suite for peak arithmetic performance measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from units import Performance

from ..benchmark import ArithmeticBenchmark
from ..generation import ArithmeticBenchmarkParams
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from context import CARMContext


@dataclass
class ArithmeticBenchmarkSuite(ISABenchmarkSuite):
    """Suite for arithmetic-only benchmarks (TestType.ARITHMETIC).

    Measures peak arithmetic performance for a specific operation.
    """

    @classmethod
    def generate(cls, context: CARMContext, isa_name: str) -> ArithmeticBenchmarkSuite:
        """Generate arithmetic microbenchmarks for a single ISA.

        Maps precision to DataType, instantiates ISA, and generates benchmark specs.

        Args:
            context: CARM context with benchmarking and architecture config.
            isa_name: Name of the ISA to generate benchmarks for (e.g., "avx2", "avx512").

        Returns:
            ArithmeticBenchmarkSuite with arithmetic benchmarks for the ISA.

        Raises:
            ValueError: Unsupported precision or unknown ISA name.
        """
        from benchmark.generation.isa import BaseISA

        benchmark = context.benchmarking
        architecture = context.architecture

        # Find ISA class by name
        isa_class = next(
            (isa_cls for isa_cls in architecture.isa if isa_cls.name == isa_name),
            None,
        )
        if isa_class is None:
            raise ValueError(f"Unknown ISA: {isa_name}")

        # Use factory method to instantiate ISA with architecture-specific parameters
        isa: BaseISA = isa_class.from_architecture(architecture)

        # Create ISA suite
        suite = cls(isa_name=isa.name)

        # Cache level doesn't matter for arithmetic benchmarks, but we still need to plan thread affinity
        # This also helps us deal with SMT, ensuring threads are properly distributed across cores
        thread_affinity = architecture.memory_topology.plan_thread_affinity(benchmark.threads, 1)

        # Generate arithmetic benchmark
        params = ArithmeticBenchmarkParams(
            data_type=benchmark.data_type,
            thread_affinity=thread_affinity.cpu_ids,
            operation=benchmark.instruction,
            num_ops=benchmark.num_ops,
        )
        spec = isa.generate_arithmetic(params, context)

        # Add to ISA suite using the benchmark's authoritative name
        arith_benchmark = ArithmeticBenchmark(params=params, spec=spec)
        suite.add_benchmark(arith_benchmark.name, arith_benchmark)

        return suite

    def get_peak_performance(self) -> Performance:
        """Get peak GOPS from arithmetic portion of roofline suite.

        Returns:
            Peak GOPS from arithmetic benchmarks, or None if unavailable.
        """
        arith_benchmarks = self.get_arithmetic_benchmarks()
        if len(arith_benchmarks) > 1:
            raise ValueError("Expected exactly one arithmetic benchmark for peak performance calculation")

        return super().get_peak_performance()
