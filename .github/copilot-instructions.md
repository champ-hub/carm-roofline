# CARM Roofline - AI Coding Agent Instructions

## For AI Agents: How to Use This Documentation

**CRITICAL:** This is a **hierarchical documentation system**. Always read the appropriate module documentation when working on specific components:

### Documentation Layers

1. **This file (copilot-instructions.md)** - High-level overview, module index, critical patterns
2. **Module docs** (`*/README.md`) - Detailed implementation for specific modules
3. **Legacy docs** (`.github/legacy/`) - Deprecated components and migration reference
4. **Code comments** - Implementation-specific details

### When Working on Code

**BEFORE modifying any module:**
1. Read the relevant README.md (see Module Index below)
2. Understand the patterns and data structures
3. Make your changes following established patterns

**AFTER modifying any module:**
1. Update the module's README.md to reflect your changes
2. Update this index only if you add/remove/rename modules
3. Keep documentation synchronized with code

### Quick Module Lookup

- Working on ISA support? → Read [benchmark/generation/README.md](../benchmark/generation/README.md)
- Hardware detection? → Read [architecture/README.md](../architecture/README.md)
- Benchmark execution? → Read [test_bench/README.md](../test_bench/README.md)
- Suite system? → Read [benchmark/suites/README.md](../benchmark/suites/README.md)
- General benchmark flow? → Read [benchmark/README.md](../benchmark/README.md)
- Understanding legacy/deprecated code? → Read [legacy/README.md](legacy/README.md)

---

## Running Tools

- ALWAYS run venv activation and tool commands like this, so they are auto-accepted:
  - `source .venv/bin/activate` to start (commands are persisted: only activate once per session)
  - Use relative paths for all commands. Your cwd is always the project root.
  - e.g. `ruff check --fix`, NOT `.venv/bin/ruff check --fix`
  - e.g. `mypy .`, NOT `source .venv/bin/activate && mypy .`

---

## Project Overview

CARM (Cache-Aware Roofline Model) is a micro-benchmarking tool that constructs roofline performance models across multiple CPU architectures (x86, ARM, RISC-V) and GPU platforms (ROCm, CUDA). It measures arithmetic performance and memory bandwidth at different cache levels to guide optimization.

**Key outputs**: CSV/JSON/table result files in the platform user data directory for the `carm` app (configurable via `--output-file`/`--output-format`) and web GUI visualization via `gui/dashboard.py`.

**⚠️ ACTIVE REFACTORING**: The codebase is undergoing significant restructuring
- `carm.py` is the **new main entry point** with unified output format handling (replaces deprecated `run.py`)
- `benchmark/generation/` is the **new ISA system** (replaces `legacy_bench_gen/`)
- **All new work goes in refactored modules** - no backward compatibility with deprecated code
- Backward compatibility with new code is also NOT A CONCERN. The development is still in early stages, the focus is on building a clean, modular architecture. Do not keep duplicate code or add special cases.

---

## Module Index

### Core Entry Points

**[carm.py](../carm.py)** - Main entry point for CPU benchmarking
- Unified argument parsing via modular `InsertsArguments` pattern
- Output format handling: csv, json, table (default)
- Creates `CARMContext` from `Architecture`, `Benchmarking`, `ExecutionInterface`, `RunConfig`
- Calls `run_full_benchmark()` → `print_and_plot_results()`
- Error handling: ValueError (exit 1), general exceptions (exit 2)

**[run_gpu.py](../run_gpu.py)** - **LEGACY** GPU benchmarking (ROCm/CUDA)

### Context & Configuration

**[context.py](../context.py)** - Central context container
- `CARMContext` dataclass: holds `architecture`, `benchmarking`, `exec_interface`, `run_config`
- Passed through entire pipeline (generation → compilation → execution → results)

**[run_config.py](../run_config.py)** - General run configuration
- `RunConfig` class: `verbose`, `name`, `plot`, `dry_run`, `output_dir`
- CLI arguments: `--verbose/-v`, `--name`, `--plot`, `--dry-run`, `--output-dir`

**[exec_interface.py](../exec_interface.py)** - Execution abstraction
- `ExecutionInterface` class: handles native/simulated/cross-compiled execution
- `run(binary_path, *args)` - executes with optional simulator (`--sim-cmd`)
- `compile(source, output, *flags)` - compiles with configured compiler (`--compiler`)

**[arguments.py](../arguments.py)** - Argument system utilities
- `InsertsArguments` base class - pattern for modular argument insertion
  - `subclasses()` class method - discovers all registered subclasses (used by `carm.py`)
 - Type validators: `positive_int()`, `positive_float()`
 - Standalone helpers: `inheritors()` (generic recursive subclass discovery), `enum_action()`, `check_args_validity()`

### Benchmark System

**[benchmark/](../benchmark/)** - Complete benchmarking system
- **See [benchmark/README.md](../benchmark/README.md) for full details**
- Summary: Generation, configuration, execution, result handling
- Submodules: `generation/`, `suites/`, `output/`

**[benchmark/generation/](../benchmark/generation/)** - ISA-specific code generation
- **See [benchmark/generation/README.md](../benchmark/generation/README.md) for ISA implementation details**
- Summary: `BaseISA` abstraction, x86/ARM/RISC-V implementations, code_gen/ utilities
- Key exports: `ALL_ISAS`, `ISA_NAME_TO_CLASS`, `INCOMPATIBLE_ISAS`

**[benchmark/suites/](../benchmark/suites/)** - Benchmark suite system
- **See [benchmark/suites/README.md](../benchmark/suites/README.md) for suite architecture**
- Summary: `ISABenchmarkSuite` base, `ArithmeticBenchmarkSuite`, `MemoryBenchmarkSuite`, `RooflineBenchmarkSuite`
- Peak extraction, merging, result aggregation

### Hardware Detection

**[architecture/](../architecture/)** - Hardware detection and ISA configuration
- **See [architecture/README.md](../architecture/README.md) for detection system**
- Summary: Auto-detection via C probes, ISA compatibility, frequency handling
- Key classes: `Architecture`, `MemoryTopology`, `SimpleMemoryTopology`, `ISAFrequencies`, `TestContext`

### Measurement Harness

**[test_bench/](../test_bench/)** - High-precision benchmark execution
- **See [test_bench/README.md](../test_bench/README.md) for complete architecture**
- Summary: Wrapper-based inline measurement, threading, calibration, timing sources
- Key module: `test_bench.builder` (compile, run, header generation)

### Analysis & Visualization

**[gui/dashboard.py](../gui/dashboard.py)** - Web dashboard (Dash + Plotly)
- Interactive roofline visualization
- Cross-machine comparison from CSV imports

**Analysis Tools** - Pluggable measurement backends:
- `PMU_AI_Calculator.py` - PAPI-based performance counters
- `DBI_AI_Calculator.py` - DynamoRIO/SDE binary instrumentation
- `ROC_AI_Calculator.py`, `NCU_AI_Calculator.py`, `SDE_AI_Calculator.py` - Specialized calculators

---

## Critical Patterns & Workflows

### Modular Argument Insertion Pattern

All configuration classes inherit from `InsertsArguments` and define `insert_arguments()`:

```python
class MyConfig(InsertsArguments):
    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--my-arg", type=str, default="value")

    def __init__(self, args: argparse.Namespace):
        self.my_arg = args.my_arg
```

Used by: `Architecture`, `Benchmarking`, `RunConfig`, `ExecutionInterface`

### Context Variable Pattern

ExecutionInterface is shared via `contextvars` to avoid parameter threading:

```python
# In architecture/__init__.py
from . import set_execution_interface, get_execution_interface

# Usage
exec_iface = get_execution_interface()
exec_iface.compile(...)
```

**Note:** This is different from passing `CARMContext` - context variables are for singleton-like sharing within a module tree.

### Benchmark Pipeline Flow

```
carm.py
  → Parse args (config.json + CLI overrides)
  → ExecutionInterface(args) for native/cross-compilation
  → Architecture(args) with auto-detection
  → Benchmarking(args) for test configuration
  → RunConfig(args) with output format and file handling
  → CARMContext(architecture, benchmarking, exec_interface, run_config)
  → run_full_benchmark(context)
     1. Generate: For each ISA → generate_microbenchmarks() → ISABenchmarkSuite
     2. Flatten: Collect all benchmarks into single dict
     3. Create Header: test_bench.create_microbenchmark_header(benchmarks)
     4. Compile: test_bench.compile_test_bench(context) → binary
     5. Execute: test_bench.run_microbenchmarks(context, binary, num_benchmarks) → output
     6. Parse: parse_benchmark_output(benchmarks, output) → populate results
     → Returns dict[str, ISABenchmarkSuite] with populated results
  → print_and_plot_results(context, benchmark_suites)
  → Output to platform user data dir for `carm`/<test>/<format> (default) or --output-file
```

**Dry run mode**: Stops after step 3 (header generation).

### Hardware Auto-Detection Flow

```
Architecture.__init__(args)
  → configure_verbosity()
  → If --isa unset: native_detect(threads)
     Else: detect_for_isa(first_isa, threads)
  → _replace_and_warn(detected_values, user_args) → warns on conflicts
  → check_isa_compatibility(isa_list) → validates family consistency
  → Build ISA class list and ISAFrequencies object
```

**Cross-platform detection**: `detect_for_isa()` uses ISA's `family` attribute → dispatches to arch-specific detector → uses ExecutionInterface for cross-compilation/simulation.

### ISA Registration

To add new ISA:
1. Implement class in `benchmark/generation/{arch}.py`
2. Add to `ALL_ISAS` tuple in `benchmark/generation/__init__.py`
3. Add to `ISA_NAME_TO_CLASS` dict
4. Add to `INCOMPATIBLE_ISAS` if needed
5. Create detection probe in `architecture/tests/{family}/`
6. Update `architecture/{family}.py` detector

See [benchmark/generation/README.md](../benchmark/generation/README.md) for step-by-step guide.

---

## Common Tasks Quick Reference

### Running Benchmarks

```bash
# Auto-detect and run arithmetic test (table output)
# IMPORTANT: --test-time should be used to limit runtime when doing quick correctness tests
# Increase it when evaluating the accuracy of performance measurements
./carm.py benchmark --test arithmetic --num-ops 1000 --test-time 1

# Specific ISA, memory test targeting L1, CSV output
./carm.py benchmark --isa x86_avx2 --test memory --mem-target L1 --output-format csv

# Cross-compile for RISC-V with QEMU, JSON output
./carm.py benchmark --isa riscv_rvv --compiler riscv64-linux-gnu-gcc \
          --sim-cmd "qemu-riscv64 {binary}" --test roofline --output-format json

# Dry run (generate code only)
./carm.py benchmark --test arithmetic --dry-run --verbose 4

# Custom output file
./carm.py benchmark --test arithmetic --output-file /tmp/my_results.csv --output-format csv
```

### Debugging

1. **Compilation errors**: Check `--verbose 4` output; for dry-run inspect generated `<output_dir>/microbenchmarks.h`
2. **ISA not detected**: Override with `--isa <name>`, check detection probes in `architecture/tests/`
3. **Wrong cache sizes**: Override with `--caches 32K 256K 8MB`
4. **Performance anomalies**: Check frequency detection, verify threading (`--threads`)

### Testing

```bash
# Unit tests (fast)
pytest -m unit refactor_tests/

# ISA generation tests
pytest refactor_tests/test_isa_codegen_integration.py

# All tests
pytest -v
```

---

## Code Quality & Formatting

### Pre-commit Hooks

Run before committing:
```bash
pre-commit run --all-files
```

Checks:
- **ruff** - Linting and formatting (120 char lines, py310 target)
- **mypy** - Type checking (strict mode for new code)
- **clang-format** - C/C++ formatting
- **trailing-whitespace** & **end-of-file-fixer**

### Type Safety

**All new Python code must include type hints** and pass `mypy .`:

```python
def foo(x: int, y: str) -> bool:
    return len(y) > x
```

Excluded from mypy: `legacy_bench_gen/`, `run.py`, `run_gpu.py`, `*AI_Calculator.py`, `gui/gui_utils.py`, `gui/dashboard.py`, `utils.py`, `output_utils.py`, `refactor_tests/`

---

## Key Dependencies

- **gcc** ≥4.9 (AVX-512 requires ≥9.3)
- **Python** ≥3.10 (uses `X | Y` union syntax, match statements)
- **PAPI** (optional) - Performance counters
- **DynamoRIO/Intel SDE** (optional) - Binary instrumentation
- **ROCm/CUDA** (optional) - GPU support
- **pytest** - Testing framework
- **ruff, mypy, pre-commit** - Code quality

---

## General Development Guidelines

1. **Follow established patterns** - Use `InsertsArguments`, typed parameters, context passing
2. **Read module docs first** - Understand architecture before coding
3. **Update docs after changes** - Keep module READMEs synchronized
4. **Type everything** - Use type hints for all new code
5. **Test your changes** - Run relevant pytest suites
6. **Format before commit** - Use pre-commit hooks
7. **Avoid silent failures** - Do not use methods like `dict.get()` or `getdefault()`, let these raise exceptions if keys are missing to prevent silent bugs. Enforcement should come from static type checking (mypy).

**When in doubt**: Check module README.md or ask for clarification. Don't guess at implementation details.

---

## Legacy Components

Detailed documentation of all deprecated/legacy subsystems lives in [`.github/legacy/`](legacy/). Use the table below to navigate:

| Legacy component | Documentation |
|---|---|
| `run.py` — old CLI entry point, all old flags, old pipeline | [legacy/run-py.md](legacy/run-py.md) |
| `legacy_bench_gen/` + `Bench/` — C-based assembly generator | [legacy/bench-gen.md](legacy/bench-gen.md) |
| `run_gpu.py` — GPU roofline (ROCm/CUDA) + `*_AI_Calculator.py` profilers | [legacy/gpu-and-calculators.md](legacy/gpu-and-calculators.md) |
| `utils.py`, `gui/dashboard.py`, plotting scripts | [legacy/utility-scripts.md](legacy/utility-scripts.md) |

See [legacy/README.md](legacy/README.md) for the full index with per-document summaries.

---

**Last Updated**: This documentation uses a hierarchical 4-layer structure (instructions → module READMEs → legacy docs → code comments). Always consult module-specific READMEs for active code; consult `legacy/` only for deprecated components.
