"""Arithmetic benchmark suite for peak arithmetic performance measurement."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from carm_roofline.core import UserError
from carm_roofline.output_utils import warn

from ..benchmark import ArithmeticBenchmark
from ..generation import ArithmeticBenchmarkParams
from .base import ISABenchmarkSuite

if TYPE_CHECKING:
    from carm_roofline.context import CARMContext


@dataclass
class ArithmeticBenchmarkSuite(ISABenchmarkSuite):
    """Suite for arithmetic-only benchmarks (TestType.ARITHMETIC).

    Measures peak arithmetic performance across one or more operations.
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
        from carm_roofline.isa import BaseISA

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
        # This also helps us deal with SMT, ensuring threads are properly distributed across cores.
        for thread_count in benchmark.threads:
            thread_affinity = architecture.memory_topology.plan_thread_affinity(thread_count, 1)
            for data_type, operation in product(benchmark.data_type, benchmark.instructions):
                if operation not in isa.bench_instructions.available_operations(data_type):
                    warn(
                        f"Skipping instruction '{operation.name}' for data type '{data_type.name}' on ISA "
                        f"'{isa_name}': no such instruction is available"
                    )
                    continue
                params = ArithmeticBenchmarkParams(
                    data_type=data_type,
                    thread_affinity=thread_affinity.cpu_ids,
                    operation=operation,
                    num_ops=benchmark.num_ops,
                )
                spec = isa.generate_arithmetic(params, context)

                # Add to ISA suite using the benchmark's authoritative name
                arith_benchmark = ArithmeticBenchmark(params=params, spec=spec)
                suite.add_benchmark(arith_benchmark.name, arith_benchmark)

        if not suite.benchmarks:
            raise UserError(
                f"No arithmetic benchmarks can be generated for ISA '{isa_name}' with data type(s) "
                f"{[dt.name for dt in benchmark.data_type]}: none of the requested instructions "
                "is available for any of them"
            )
        return suite
