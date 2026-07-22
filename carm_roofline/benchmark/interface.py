"""High-level benchmark orchestration and generation interface.

This module coordinates the entire benchmark pipeline:
- Generating microbenchmark specifications
- Compiling benchmark binaries
- Running benchmarks and collecting results
- Aggregating results into BenchmarkResult objects
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from carm_roofline.output_utils import debug, detail
from carm_roofline.test_bench.builder import (
    MicrobenchmarkFunctionSpec,
    compile_test_bench,
    create_microbenchmark_header,
    run_microbenchmarks,
)
from carm_roofline.workspace import workspace_context

from .benchmark import BaseBenchmark
from .result import parse_benchmark_output
from .suites import (
    ArithmeticBenchmarkSuite,
    ISABenchmarkSuite,
    MemoryBenchmarkSuite,
    MemorySweepBenchmarkSuite,
    MixedBenchmarkSuite,
    RooflineBenchmarkSuite,
)

if TYPE_CHECKING:
    from carm_roofline.context import CARMContext


def generate_microbenchmarks(context: CARMContext, isa_name: str) -> ISABenchmarkSuite:
    """Generate microbenchmarks for a single ISA, returning specification.

    Delegates to the appropriate suite class based on TestType.
    Does not write files or compile binaries.

    Args:
        context: Full CARM context with benchmark configuration.
        isa_name: Name of the ISA to generate benchmarks for (e.g., "avx2").

    Returns:

        ISABenchmarkSuite subclass instance for the specified ISA and test type.

    Raises:
        NotImplementedError: If test type is not yet implemented.
        ValueError: If ISA name is unknown or test type is invalid.
    """
    from carm_roofline.benchmark.benchmarking import TestType

    test_type: TestType = context.benchmarking.test

    # Factory dispatch: map TestType to suite class
    suite_class_map = {
        TestType.ARITHMETIC: ArithmeticBenchmarkSuite,
        TestType.MEMORY: MemoryBenchmarkSuite,
        TestType.ROOFLINE: RooflineBenchmarkSuite,
        TestType.MIXED: MixedBenchmarkSuite,
        TestType.MEMORY_SWEEP: MemorySweepBenchmarkSuite,
    }

    suite_class = suite_class_map.get(test_type)
    if suite_class is None:
        raise ValueError(f"Unknown test type: {test_type}")

    # Delegate to suite class's generate() method
    return suite_class.generate(context, isa_name)


def run_full_benchmark(
    context: CARMContext,
) -> dict[str, ISABenchmarkSuite]:
    """Execute complete benchmark pipeline: generate → compile → run → aggregate.

    This is the high-level orchestrator function that coordinates all steps
    of the benchmark process for all configured ISAs, returning structured
    BenchmarkResult objects grouped by ISA.

    Optimization: Generates microbenchmarks for all ISAs, then compiles and runs
    a single binary with all benchmarks. This is much faster than per-ISA compilation.

    Args:
        context: Full CARM context with architecture, benchmark, and execution config.

    Returns:
        Dictionary mapping ISA names to ISABenchmarkSuite objects with results populated.

        If `context.run_config.dry_run` is True, benchmarks are generated and
        the header file is written, but compilation and execution are skipped.

    Raises:
        RuntimeError: If any step (generation, compilation, or execution) fails.
    """
    # Step 1: Generate microbenchmarks for all ISAs (loop over ISAs)
    isa_suites: dict[str, ISABenchmarkSuite] = {}

    for isa_class in context.architecture.isa:
        isa_name = isa_class.name
        debug(f"Generating benchmarks for {isa_name}...")
        suite = generate_microbenchmarks(context, isa_name)
        isa_suites[isa_name] = suite
        debug(f"  Generated {len(suite.benchmarks)} benchmark(s)")

    total_benchmarks = sum(len(suite.benchmarks) for suite in isa_suites.values())
    detail(f"Generated {total_benchmarks} benchmark(s) across {len(isa_suites)} ISA(s)")

    # Step 2: Flatten all ISA suites into a single dict (for compilation)
    flat_benchmarks: dict[str, BaseBenchmark] = {}
    for _isa_name, suite in isa_suites.items():
        for bench_name, benchmark in suite.benchmarks.items():
            flat_benchmarks[bench_name] = benchmark

    debug(f"Flattened benchmarks for compilation: {list(flat_benchmarks.keys())}")

    keep_workspace = context.run_config.dry_run or context.run_config.keep_artifacts
    with workspace_context(keep=keep_workspace, prefix="carm-benchmark-") as workspace_dir:
        workspace = Path(workspace_dir)

        if keep_workspace:
            detail(f"Artifacts will be kept in: {workspace}")

        # Keep generated artifacts in a writable temporary workspace.
        # Static C sources remain in-package and are included explicitly at compile time.
        generated_header = workspace / "microbenchmarks.h"
        generated_binary = workspace / "benchmark"

        # Step 3: Create header file from all benchmark specifications
        create_microbenchmark_header(flat_benchmarks.values(), generated_header)

        if context.run_config.dry_run:
            detail(f"Dry run: wrote generated header to {generated_header}; skipping compilation and execution.")
            return isa_suites

        # Step 4: Compile single binary with all benchmarks
        detail("Compiling benchmark...")
        binary_path = compile_test_bench(
            context,
            output_path=generated_binary,
            include_dirs=(workspace,),
        )
        debug(f"Compiled binary: {binary_path}")

        # Step 5: Run compiled binary (all ISAs and benchmarks at once)
        expected_runtime = context.benchmarking.test_time * total_benchmarks
        detail(f"Running benchmark. Expected runtime: >{expected_runtime:.0f} seconds...")
        raw_output = run_microbenchmarks(context, binary_path, (b.spec for b in flat_benchmarks.values()))

        # Step 6: Parse results and populate BenchmarkResult objects
        parse_benchmark_output(flat_benchmarks, raw_output)

    return isa_suites


__all__ = [
    "MicrobenchmarkFunctionSpec",
    "generate_microbenchmarks",
    "run_full_benchmark",
]
