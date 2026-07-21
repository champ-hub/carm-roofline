"""Output module for CARM benchmarks.
Dispatches output kinds to test-specific output handlers.
"""

from __future__ import annotations

from benchmark.benchmark import ISABenchmarkSuite
from benchmark.benchmarking import TestType
from context import CARMContext

from .base import OutputHandler, OutputKind

__all__ = [
    "OutputKind",
    "TestType",
    "_get_handler_for_test_type",
    "output_benchmark_results",
]


def _strategy_registry() -> dict[TestType, type[OutputHandler]]:
    from . import arithmetic, memory, memory_sweep, mixed, roofline

    return {
        TestType.ARITHMETIC: arithmetic.ArithmeticOutputHandler,
        TestType.ROOFLINE: roofline.RooflineOutputHandler,
        TestType.MEMORY: memory.MemoryOutputHandler,
        TestType.MIXED: mixed.MixedOutputHandler,
        TestType.MEMORY_SWEEP: memory_sweep.MemorySweepOutputHandler,
    }


def _get_handler_for_test_type(test_type: TestType) -> OutputHandler:
    registry = _strategy_registry()
    try:
        strategy_class = registry[test_type]
    except KeyError as exc:
        supported = ", ".join(t.value for t in sorted(registry, key=lambda item: item.value))
        raise ValueError(f"Unsupported or unregistered test type: {test_type!r}. Supported: {supported}") from exc

    return strategy_class()


def output_benchmark_results(context: CARMContext, isa_suites: dict[str, ISABenchmarkSuite]) -> None:
    """Utility to print CLI summary and generate plots for benchmark results.

    Args:
        context: CARMContext containing run configuration and output preferences
        isa_suites: Benchmark results organized by ISA
    """
    output_handler = _get_handler_for_test_type(context.benchmarking.test)
    output_handler.handle(context, isa_suites)
