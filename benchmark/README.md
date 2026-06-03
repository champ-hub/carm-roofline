# Benchmark Module Documentation

This module handles benchmark generation, configuration, execution, and result processing for the CARM roofline tool.

## Module Overview

The benchmark system is organized into several key components:

- **[benchmarking.py](benchmarking.py)** - Configuration and command-line argument handling
- **[benchmark.py](benchmark.py)** - Core data structures (typed benchmarks and results)
- **[interface.py](interface.py)** - High-level orchestration and pipeline
- **[result.py](result.py)** - Result parsing and output formatting
- **[generation/](generation/README.md)** - ISA-specific code generation (see submodule docs)
- **[suites/](suites/README.md)** - Benchmark suite system (see submodule docs)
- **[output/](output/)** - Output formatting handlers

## Core Data Structures

### Typed Benchmark Hierarchy

The system uses a typed class hierarchy to represent different benchmark types:

**Base Classes:**
- `BaseBenchmark` (abstract) - Common benchmark interface
  - `name: property` - Returns `spec.function_name` (authoritative identifier)
  - `params: BenchmarkParams` - Test parameters
  - `spec: MicrobenchmarkFunctionSpec` - Generated function specification
  - `results: BaseBenchmarkResult | None` - Populated after execution

**Concrete Benchmark Types:**
- `ArithmeticBenchmark` - Pure floating-point arithmetic tests
  - Parameters: `ArithmeticBenchmarkParams` (operation, num_ops, data_type)

- `MemoryBenchmark` - Memory bandwidth tests
  - Parameters: `MemoryBenchmarkParams` (load_store_ratio, size_per_thread, memory_level_name, data_type)
  - Additional: `cache_level` to target specific cache (L1/L2/L3/DRAM)
  - Sizing policy in generation: non-final levels use 80% of per-thread available size; final level uses exactly 2x previous level per-thread size
  - Final level naming follows topology iteration and is annotated as `DRAM` when provided

- `MixedBenchmark` - Combined arithmetic + memory stress
  - Parameters: Both arithmetic and memory params
  - Additional: `arithmetic_intensity` ratio

### Typed Result Hierarchy

Results are strongly typed to match benchmark types:

**Base Class:**
- `BaseBenchmarkResult` (abstract)
  - `time_taken_ms: float` - Median runtime in milliseconds
  - `num_repetitions: int` - Calibrated repetition count

**Concrete Result Types:**
- `ArithmeticBenchmarkResult`
  - `operations: Operations` - Giga-operations per second

- `MemoryBenchmarkResult`
  - `bandwidth: Bandwidth` - Memory bandwidth
  - `cache_level: str` - Target cache level (L1/L2/L3/DRAM)

- `MixedBenchmarkResult`
  - Both gops and bandwidth metrics
  - `arithmetic_intensity: float` - FLOPs per byte ratio

**Important:** Result fields are **public** (no underscore prefix). Results are computed by `Benchmark.process_results()` after execution based on the parameter type.

## Configuration (benchmarking.py)

### Benchmarking Class

Encapsulates all benchmark configuration via the `InsertsArguments` pattern:

**Fields:**
- `test: TestType` - Type of test to run (ARITHMETIC, MEMORY, ROOFLINE, MIXED)
- `mem_target: str` - Memory hierarchy target (L1/L2/L3/DRAM)
- `precision: DataType` - f32 or f64
- `threads: int` - Number of threads to use
- `interleaved: bool` - NUMA interleaved thread mapping
- `instructions: set[ArithmeticOperation]` - Arithmetic operations to test (default: add, fma)
- `num_ops: int` - Number of operations per benchmark
- `ld_st_ratio: LoadStoreRatio` - Load/store ratio for memory tests
- `arith_mem_ratio: tuple[int, int] | None` - Arithmetic/memory ratio for mixed tests
- `mem_test_sizes: list[Bytes | None] | None` - Per-level memory test sizes as Bytes objects, or None for automatic sizing
- `test_time: float` - Target runtime per benchmark (seconds)

**TestType Enum:**
```python
class TestType(Enum):
    ARITHMETIC = "arithmetic"
    MEMORY = "memory"
    ROOFLINE = "roofline"
    MIXED = "mixed"

    # Shortened aliases
    ARITH = ARITHMETIC
    MEM = MEMORY
    ROOF = ROOFLINE
    MIX = MIXED
```

**LoadStoreRatio Class:**
Encapsulates load:store ratio parsing (e.g., "2:1" → 2 loads, 1 store).

**Type Validators:**
- `test_type(arg)` - Converts string to TestType enum
- `precision_type(arg)` - Validates precision (f32/f64)
- `ld_st_ratio_type(arg)` - Parses load:store ratio strings

**Warning:** Test time < 10s triggers a warning about potentially unreliable results.

### Command-Line Arguments

Via `insert_arguments()` static method:
- `--test/-t` - Test type (arithmetic/memory/roofline/mixed)
- `--mem-target` - Cache level target (L1/L2/L3/DRAM)
- `--precision/-p` - f32 or f64
- `--threads` - Number of threads
- `--interleaved` - Enable NUMA interleaved mapping
- `--instruction` - One or more arithmetic instructions to test (space-separated; default: add fma)
- `--num_ops` - Operations per benchmark
- `--ld_st_ratio` - Load:store ratio (e.g., "2:1")
- `--arith_mem_ratio` - Arithmetic:memory ratio for mixed tests
- `--mem_test_sizes` - Memory test sizes
- `--test-time` - Target runtime per benchmark (default: 10s)

## Pipeline Orchestration (interface.py)

### generate_microbenchmarks()

**Signature:** `generate_microbenchmarks(context: CARMContext, isa_name: str) -> ISABenchmarkSuite`

Generates benchmark specifications for a single ISA by delegating to suite classes based on TestType:

```python
suite_class_map = {
    TestType.ARITHMETIC: ArithmeticBenchmarkSuite,
    TestType.MEMORY: MemoryBenchmarkSuite,
    TestType.ROOFLINE: RooflineBenchmarkSuite,
    TestType.MIXED: MixedBenchmarkSuite,
}
```

**Returns:** Suite object (see [suites/README.md](suites/README.md)) containing generated benchmarks.

### run_full_benchmark()

**Signature:** `run_full_benchmark(context: CARMContext) -> dict[str, ISABenchmarkSuite]`

Complete benchmark pipeline with multi-ISA support:

**6-Step Process:**

1. **Generate** - Loop over ISAs, call `generate_microbenchmarks(context, isa_name)` for each
   - Returns `dict[str, ISABenchmarkSuite]` mapping ISA name → suite

2. **Flatten** - Collect all benchmarks from all ISA suites into single dict
   - `{benchmark.name: benchmark}` for all benchmarks across all suites

3. **Create Header** - Call `test_bench.create_microbenchmark_header(benchmarks, output_path=...)`
  - Generates `microbenchmarks.h` in a temporary writable workspace
  - Single header file contains ALL ISAs' benchmarks

4. **Compile** - Call `test_bench.compile_test_bench(context, source_path=..., output_path=...)`
   - Compiles single binary containing all benchmarks
   - Uses ExecutionInterface for cross-compilation support
   - Raises `RuntimeError` on compilation failure

5. **Execute** - Call `test_bench.run_microbenchmarks(context, binary_path, specs)`
   - `specs` is an iterable of `MicrobenchmarkFunctionSpec` (one per benchmark)
   - Timeout is derived from each spec's `read_array_size` / `write_array_size` / `num_threads`
     using a 1 GB/s bandwidth floor and 10× safety factor (physics-based, not heuristic)
   - Runs binary once (executes all benchmarks)
   - Returns raw stdout output
   - Raises `RuntimeError` on execution failure

6. **Parse** - Call `parse_benchmark_output(flat_benchmarks, raw_output)`
   - Parses output and populates `Benchmark.results` for each benchmark
   - Computes derived metrics (gops, bandwidth, IPC)

Note: Output format and file location are now owned by `context.run_config` and applied in the output module (`print_and_plot_results`), not during benchmark execution.

**Dry Run Mode:** If `context.run_config.dry_run` is True, skips steps 4-6 (only generates header).

**Returns:** `dict[str, ISABenchmarkSuite]` with all results populated.

**Integration with test_bench:**
```python
from test_bench.builder import (
    create_microbenchmark_header,
    compile_test_bench,
    run_microbenchmarks,
)
```

## Result Processing (result.py)

### OutputFormat Enum

Supported output formats:
- `CSV` - Comma-separated values
- `JSON` - JSON object
- `TABLE` - Human-readable table
- `JSONL` - JSON Lines (one object per line)

### parse_benchmark_output()

**Signature:** `parse_benchmark_output(benchmarks: dict[str, Benchmark], raw_output: str) -> None`

Parses test_bench output and populates `Benchmark.results` fields:

**Input Format (per benchmark):**
```
benchmark_name, runtime_ms, num_repetitions
```

**Processing:**
1. Split output into lines
2. For each line, extract benchmark name, runtime, num_reps via `_parse_benchmark_result_line()`
3. Look up benchmark in dict by name
4. Create appropriate `BenchmarkResult` object based on benchmark type
5. Call `benchmark.process_results()` to compute derived metrics

**Derived Metrics (computed by `Benchmark.process_results()`):**
- **ArithmeticBenchmark:** `gops`, `instructions_per_cycle`
- **MemoryBenchmark:** `bandwidth_gb_per_sec` (based on cache_level size)
- **MixedBenchmark:** Both gops and bandwidth, plus `arithmetic_intensity`

### _parse_benchmark_result_line()

**Signature:** `_parse_benchmark_result_line(line: str) -> tuple[str, float, int]`

Parses individual result line into (name, runtime_ms, num_repetitions).

**Error Handling:** Raises `ValueError` if line format is invalid.

## Output Formatting (output/)

The `output/` package (not a single file) handles result presentation:

### Strategy-Based Dispatch

- `print_and_plot_results(context, benchmark_suites)` is the output entrypoint called from `carm.py`.
- `_create_output_handler(test_type, formatter_type)` selects either `CLIOutputFormatter` or `PlotterOutputFormatter`.
- Formatter dispatch is keyed by `TestType` for both paths:
  - CLI path calls `format_and_print(context, isa_suites)`
  - Plot path calls `plot_and_save(isa_suites, output_dir)`
- Handler modules provide strategy wrappers with the same method surface:
  - `ArithmeticOutputStrategy`
  - `MemoryOutputStrategy`
  - `MemorySweepOutputStrategy`
  - `MixedOutputStrategy`
  - `RooflineOutputStrategy`

These strategy wrappers delegate to module-level handler functions, keeping behavior stable while exposing a consistent strategy API.

### Legacy Roofline CSV Compatibility Mode

- Trigger condition: `context.benchmarking.test == TestType.ROOFLINE` and `OutputKind.CSV in context.run_config.output_formats`.
- Output location: `<output-dir>/roofline/<run-name>_roofline.csv`.
- Compatibility intent: two-header format to match legacy `run.py` roofline CSV layout.
  - Secondary header row: `Name:, <name>, L1 Size:, <l1>, L2 Size:, <l2>, L3 Size:, <l3>, '', L1, L1, L2, L2, L3, L3, DRAM, DRAM, FP, FP, FP FMA, FP_FMA`
  - Primary header row: `Date, ISA, Precision, Threads, Loads, Stores, Interleaved, DRAM Bytes, FP Inst., GB/s, I/Cycle, GB/s, I/Cycle, GB/s, I/Cycle, GB/s, I/Cycle, Gflop/s, I/Cycle, Gflop/s, I/Cycle`.
- Append semantics: first call writes both headers and row; subsequent calls append row only.
- Current limitation: `FP_FMA` and `I/Cycle` fields are zero-filled in refactored path when an FMA metric is unavailable.

**Structure:**
- `__init__.py` - Exports `print_and_plot_results()` utility and `_create_output_handler()` factory
- `base.py` - `OutputKind` enum and `OutputHandler` base class
- `formatter.py` - `CLIOutputFormatter` and `PlotterOutputFormatter` classes
- `handlers/` - Test-specific handlers and strategy wrappers (`*OutputStrategy`)

**print_and_plot_results():**

Called from carm.py after benchmarking completes:
```python
print_and_plot_results(context, benchmark_suites)
```

Performs CLI output, optional legacy roofline CSV emission (`--output-format csv` + roofline test), then plotting.

## Integration with Other Modules

### Dependencies
- **[generation/](generation/README.md)** - ISA-specific code generation
- **[suites/](suites/README.md)** - Suite system for organizing benchmarks
- **[test_bench/](../test_bench/README.md)** - Measurement harness
- **[architecture/](../architecture/README.md)** - Hardware configuration
- **context.py** - CARMContext container
- **exec_interface.py** - Execution abstraction

### Used By
- **carm.py** - Main entry point calls `run_full_benchmark()`

## Common Workflows

### Adding a New Test Type

1. Add enum value to `TestType` in benchmarking.py
2. Create new suite class in `benchmark/suites/` (inherit from `ISABenchmarkSuite`)
3. Implement `generate()` class method
4. Add to `suite_class_map` in interface.py
5. Update argument handling in benchmarking.py if needed

### Adding a New Operation

1. Add operation to `ArithmeticOperation` or `MemoryOperation` enums in `generation/code_gen/operation.py`
2. Implement operation in ISA classes (x86.py, arm.py, riscv.py)
3. Update `operation_counts` dict in `generation/code_gen/operation.py` if operation count differs

### Debugging Failed Benchmarks

1. Check `--verbose 4` output for compilation/execution details
2. Use `--dry-run` and examine generated `microbenchmarks.h` for correctness
3. Re-run with `-v 4` to inspect temporary workspace compile/debug logs
4. Check stderr for test_bench error messages
5. Enable dry-run mode (`--dry-run`) to generate code without executing

## Error Handling

- **BenchParamError** - Invalid benchmark parameters (from parameters.py)
- **RuntimeError** - Compilation or execution failure
- **ValueError** - Result parsing errors

All compilation/execution errors include stderr output when available.

## See Also

- **[generation/README.md](generation/README.md)** - ISA code generation system
- **[suites/README.md](suites/README.md)** - Benchmark suite architecture
- **[../test_bench/README.md](../test_bench/README.md)** - Measurement harness details
- **[../architecture/README.md](../architecture/README.md)** - Hardware detection

---

**When modifying this module:** Update this documentation to reflect changes in interfaces, data structures, or workflows.
