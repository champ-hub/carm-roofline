"""Code generation utilities for the test_bench module.

This module provides functionality to generate microbenchmarks.h from a list
of microbenchmark function specifications. The generated header includes:
- Microbenchmark function definitions
- Wrapper instantiations for timing infrastructure
- MICROBENCHMARK_LIST macro for test registration

Also provides compilation and execution utilities for benchmark binaries.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from carm_roofline.core import Bandwidth, Bytes, Frequency, Seconds

if TYPE_CHECKING:
    from carm_roofline.benchmark import BaseBenchmark
    from carm_roofline.context import CARMContext

from carm_roofline.output_utils import debug, detail

HEADER_GUARD = "MICROBENCHMARKS_H"
TEST_BENCH_ROOT = Path(__file__).parent
DEFAULT_BENCHMARK_SOURCE = TEST_BENCH_ROOT / "test_bench.c"
DEFAULT_BENCHMARK_BINARY = Path("benchmark")
DEFAULT_TARGET_RUNTIME_MS = 100

# Physics-based timeout constants for run_microbenchmarks().
# Memory benchmarks are timed by how long it takes to transfer their working set;
# a 1 GB/s floor covers NUMA cold-fault storms, MADV_DONTNEED page faults, and
# QEMU simulation slowdown.  A 10x safety factor is applied on top.
MIN_BANDWIDTH: Bandwidth = Bandwidth(1e9)  # 1 GB/s — conservative DRAM floor
TIMEOUT_SAFETY_FACTOR: float = 10.0  # applied to summed per-benchmark estimates
TIMEOUT_MIN: Seconds = Seconds(60.0)  # absolute minimum regardless of estimates
ARITHMETIC_FALLBACK_PER_RUN: Seconds = Seconds(0.1)  # 100 ms/run — matches calibration target
# The C wrapper always begins calibration with CARM_BENCH_START_REPS (= 100) repetitions in the
# very first function call.  For large DRAM working sets this single call can transfer
# START_REPS x working_set bytes, completely dominating the measurement budget when num_runs is
# small (e.g. test_time=1 → num_runs=10).  Mirror the C constant here so the timeout formula
# explicitly reserves time for that first calibration iteration.
CALIBRATION_START_REPS: int = 10  # must match CARM_BENCH_START_REPS in test_bench.h


@dataclass
class MicrobenchmarkFunctionSpec:
    """Represents a generated microbenchmark function and its metadata.

    Each benchmark carries complete configuration for independent execution,
    including ISA-specific frequency, cache topology info, and thread counts.
    """

    function_name: str
    body: str
    read_array_size: Bytes  # Working set size for the read (load) buffer
    write_array_size: Bytes  # Working set size for the write (store) buffer
    frequency: Frequency  # ISA peak/turbo frequency (used for IPC computation)
    thread_affinity: list[int]  # Specific CPU IDs to run the benchmark on
    nominal_frequency: Frequency | None = None  # TSC base frequency for timing on x86 (None → use frequency)

    @property
    def num_threads(self) -> int:
        return len(self.thread_affinity)

    def measurement_key(self) -> tuple[object, ...]:
        """Identity of the physical experiment this spec performs.

        Two specs with equal keys render identical header entries (modulo the
        function name) and therefore measure identical work. Every field that the
        generated header consumes participates; function_name is excluded because
        it differs by construction between duplicates.
        """
        # body embeds the function name once, in its first line (the C signature),
        # so strip the first occurrence only.
        canonical_body = self.body.replace(self.function_name, "", 1)
        return (
            canonical_body,
            self.read_array_size.value,
            self.write_array_size.value,
            tuple(self.thread_affinity),
            self.frequency.value,
            None if self.nominal_frequency is None else self.nominal_frequency.value,
        )


def _validate_specs(functions: list[MicrobenchmarkFunctionSpec]) -> None:
    """Ensure specs are well-formed before rendering."""

    seen_names: set[str] = set()

    for fn in functions:
        if fn.function_name in seen_names:
            raise ValueError(f"Duplicate function_name detected: {fn.function_name}")
        seen_names.add(fn.function_name)


def render_microbenchmark_header(
    functions: Iterable[MicrobenchmarkFunctionSpec], include_guard: str = HEADER_GUARD
) -> str:
    """Render the microbenchmark header as a string.

    Args:
        functions: Iterable of microbenchmark function specifications.
        include_guard: Header guard name. Defaults to "MICROBENCHMARKS_H".

    Returns:
        The rendered header as a single string.
    """

    fn_list = list(functions)
    if not fn_list:
        raise ValueError("At least one microbenchmark function is required to render the header")

    _validate_specs(fn_list)

    # Each function's body is the complete function definition, just concatenate them
    functions_code = "\n\n".join(fn.body.rstrip() for fn in fn_list)

    # Generate metadata struct instances for each benchmark
    metadata_structs = []
    for fn_spec in fn_list:
        metadata_name = f"metadata_{fn_spec.function_name}"
        affinity_name = f"thread_affinity_{fn_spec.function_name}"
        affinity_str = "{" + ", ".join(str(cpu_id) for cpu_id in fn_spec.thread_affinity) + "}"
        # Use nominal_frequency for timing when available (x86 TSC ticks at base/nominal rate,
        # not at the turbo/peak frequency stored in .frequency).
        timing_freq = fn_spec.nominal_frequency if fn_spec.nominal_frequency is not None else fn_spec.frequency
        metadata_structs.append(
            f"static int {affinity_name}[] = {affinity_str};\n"
            f"static const benchmark_metadata_t {metadata_name} = {{\n"
            f'    .name = "{fn_spec.function_name}",\n'
            f"    .frequency_ghz = {timing_freq.as_gigahertz()}f,\n"
            f"    .thread_affinity = {affinity_name},\n"
            f"    .num_threads = {fn_spec.num_threads},\n"
            f"    .read_array_size_bytes = {fn_spec.read_array_size.value}ULL,\n"
            f"    .write_array_size_bytes = {fn_spec.write_array_size.value}ULL\n"
            f"}};"
        )
    metadata_code = "\n\n".join(metadata_structs)

    # Generate wrapper instantiations: define UBENCH_NAME, include wrapper.inl, undef
    wrapper_instantiations = []
    for fn_spec in fn_list:
        wrapper_instantiations.append(
            f'#define UBENCH_NAME {fn_spec.function_name}\n#include "wrapper.inl"\n#undef UBENCH_NAME'
        )
    wrapper_code = "\n\n".join(wrapper_instantiations)

    # Generate MICROBENCHMARK_LIST macro with X-macro entries (wrapper, metadata pointer)
    microbenchmark_list_lines = []
    for fn_spec in fn_list:
        wrapper_name = f"wrapper_{fn_spec.function_name}"
        metadata_ptr = f"&metadata_{fn_spec.function_name}"
        microbenchmark_list_lines.append(f"    X({wrapper_name}, {metadata_ptr}) \\")
    microbenchmark_list_macro = "\n".join(microbenchmark_list_lines)

    return (
        f"#ifndef {include_guard}\n"
        f"#define {include_guard}\n\n"
        '#include "test_bench.h"\n\n'
        "/* This file is auto-generated. Do not edit manually. */\n\n"
        f"{functions_code}\n\n"
        f"{metadata_code}\n\n"
        f"{wrapper_code}\n\n"
        "#define MICROBENCHMARK_LIST \\\n"
        f"{microbenchmark_list_macro}\n\n"
        f"#endif /* {include_guard} */\n"
    )


def create_microbenchmark_header(
    benchmarks: Iterable[BaseBenchmark],
    output_path: Path,
    include_guard: str = HEADER_GUARD,
) -> Path:
    """Generate microbenchmarks.h from function specifications.

    This is the primary interface for generating the microbenchmark header file.
    It renders the header content and writes it to the test_bench directory.

    Args:
        benchmarks: Iterable of Benchmark objects containing specifications.
        output_path: Destination path for the generated header.
                     Defaults to test_bench/microbenchmarks.h.
        include_guard: Optional include guard override.

    Returns:
        Path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    specs = [bench.spec for bench in benchmarks]
    content = render_microbenchmark_header(specs, include_guard=include_guard)
    debug(f"Writing microbenchmark header to {output_path}")
    output_path.write_text(content)
    return output_path


def compile_test_bench(
    context: CARMContext,
    source_path: Path = DEFAULT_BENCHMARK_SOURCE,
    output_path: Path = DEFAULT_BENCHMARK_BINARY,
    include_dirs: Iterable[Path] | None = None,
) -> Path:
    """Compile the test_bench benchmark with configuration from CARMContext.

    This function translates CARM configuration into compiler defines and uses
    the ExecutionInterface to compile the benchmark. Configuration includes:
    - Verbosity level
    - Target runtime (for calibration)
    - Number of measurement runs
    - Architecture-specific flags

    Args:
        context: CARM context with architecture, benchmark, and execution configuration.
        source_path: Path to test_bench.c source file. Defaults to test_bench/test_bench.c.
        output_path: Path for compiled binary. Defaults to test_bench/benchmark.
        include_dirs: Extra include directories for generated/runtime headers.

    Returns:
        Path to the compiled binary.

    Raises:
        RuntimeError: If compilation fails.

    Example:
        >>> from context import CARMContext
        >>> binary_path = compile_benchmark(context)
        >>> result = context.exec_interface.run(str(binary_path), capture_output=True)
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Search generated include directories first so stale in-tree headers cannot shadow
    # temporary generated files (e.g. microbenchmarks.h).
    include_paths: list[Path] = []
    if include_dirs is not None:
        include_paths.extend(include_dirs)
    include_paths.append(source_path.parent)

    target_runtime = context.benchmarking.test_time
    target_runs = int((1000 * target_runtime) // DEFAULT_TARGET_RUNTIME_MS)

    # Build compiler defines from context
    defines = [
        f"-DVERBOSITY={context.run_config.verbose}",
        f"-DNUM_RUNS={target_runs}",  # Default, can be made configurable later
    ]

    # Standard flags for benchmark compilation
    flags = [
        "-Ofast",
        "-g",
        "-pthread",  # Required for pthread barriers
        "-lm",  # Math library for ceil() in timing calculations
        "-march=native",  # Use native architecture features
        *(f"-I{include_path}" for include_path in include_paths),
        *defines,
    ]

    # Compile using the execution interface
    result = context.exec_interface.compile(
        str(source_path), str(output_path), *flags, capture_output=True, text=True, check=False
    )

    if result.returncode != 0:
        error_msg = f"Benchmark compilation failed with return code {result.returncode}"
        if result.stderr:
            error_msg += f"\nStderr:\n{result.stderr}"
        raise RuntimeError(error_msg)

    return output_path


def _compute_timeout(
    specs: list[MicrobenchmarkFunctionSpec],
    num_runs: int,
) -> Seconds:
    """Compute a physics-based execution timeout for a set of microbenchmarks.

    For memory benchmarks (non-zero working-set sizes), worst-case transfer
    time is estimated using a conservative 1 GB/s bandwidth floor.  This
    covers NUMA cold-fault storms, MADV_DONTNEED page-fault overhead, and
    QEMU/simulator slowdown.  For arithmetic benchmarks (near-zero array
    sizes) a flat 100 ms/run fallback is used (matches the C calibration
    target).

    Critically, the C calibration loop begins every benchmark with
    CARM_BENCH_START_REPS (= 100) repetitions in one function call before
    any proportional scaling occurs.  For a 512 MiB DRAM benchmark those
    100 reps transfer 100 x 512 MiB regardless of num_runs, so
    CALIBRATION_START_REPS extra equivalent runs are added to the per-
    benchmark budget to cover that first calibration iteration.

    A 10x safety factor is applied to the summed per-benchmark estimate
    before clamping to the absolute minimum.

    Args:
        specs: Benchmark specifications (read/write sizes in bytes, thread count).
        num_runs: Measurement iterations compiled into the binary (NUM_RUNS).

    Returns:
        Timeout (≥ TIMEOUT_MIN).
    """
    # effective_runs = measurement runs + worst-case first calibration iteration
    effective_runs: int = num_runs + CALIBRATION_START_REPS
    total: Seconds = Seconds(0.0)
    for spec in specs:
        working_set: Bytes = (spec.read_array_size + spec.write_array_size) * spec.num_threads
        if working_set.value > 0:
            # Time = bytes / bandwidth; sum over effective runs per benchmark.
            est = Seconds(working_set.value * effective_runs / MIN_BANDWIDTH.value)
        else:
            # Arithmetic benchmark: no memory transfer, calibration target is ~100 ms/run.
            est = ARITHMETIC_FALLBACK_PER_RUN * effective_runs
        total = total + est
    return max(TIMEOUT_MIN, total * TIMEOUT_SAFETY_FACTOR)


def run_microbenchmarks(
    context: CARMContext,
    binary_path: Path,
    specs: Iterable[MicrobenchmarkFunctionSpec],
) -> str:
    """Execute the compiled microbenchmarks and return their output.

    All benchmark configuration (frequency, thread counts, etc.) is embedded
    in the generated header at compile time. Only the interleaved flag is
    passed as a runtime argument.

    The execution timeout is derived from the physical byte-transfer cost of
    every benchmark spec.  Memory benchmarks use a conservative 1 GB/s
    bandwidth floor; arithmetic benchmarks fall back to 100 ms/run.  A
    per-benchmark calibration budget (CALIBRATION_START_REPS extra runs)
    is added before applying the 10x safety factor, covering the expensive
    first calibration iteration that always runs CARM_BENCH_START_REPS
    repetitions regardless of the configured NUM_RUNS.  This prevents
    premature timeouts when test_time is small and the working set is large.

    Args:
        context: Full CARM context with execution interface configuration.
        binary_path: Path to the compiled benchmark binary.
        specs: Microbenchmark specifications used to estimate the timeout
               (read/write array sizes and thread counts determine transfer volume).

    Returns:
        The stdout output from running the benchmarks.

    Raises:
        RuntimeError: If execution fails or binary not found.
    """
    if not binary_path.exists():
        raise RuntimeError(f"Benchmark binary not found: {binary_path}")

    spec_list = list(specs)
    num_runs = int((1000 * context.benchmarking.test_time) // DEFAULT_TARGET_RUNTIME_MS)
    timeout = _compute_timeout(spec_list, num_runs)

    debug(f"Running microbenchmarks from {binary_path}")
    debug(f"Calculated timeout: {timeout} for {len(spec_list)} benchmark(s) ({num_runs} runs each)")

    # Build argument list - only interleaved flag is runtime-configurable
    args = []
    if context.benchmarking.interleaved:
        args.append("--interleaved")

    result = context.exec_interface.run(
        str(binary_path),
        *args,
        capture_output=True,
        text=True,
        timeout=timeout.value,
        check=False,
    )

    if result.returncode != 0:
        error_msg = f"Benchmark execution failed with return code {result.returncode}\n"
        if result.stdout:
            error_msg += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            error_msg += f"stderr:\n{result.stderr}"
        raise RuntimeError(error_msg)

    if result.stderr:
        debug(f"Benchmark debug output:\n{result.stderr}")

    detail("Microbenchmarks completed successfully")
    return str(result.stdout)


__all__ = [
    "MicrobenchmarkFunctionSpec",
    "compile_test_bench",
    "create_microbenchmark_header",
    "render_microbenchmark_header",
    "run_microbenchmarks",
]
