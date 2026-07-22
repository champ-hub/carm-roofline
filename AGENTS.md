# Repository Guidelines

## Project Overview

CARM (Cache-Aware Roofline Model) is a micro-benchmarking toolkit that constructs roofline performance models across multiple CPU architectures (x86, ARM, RISC-V) and GPU platforms (ROCm, CUDA). It measures arithmetic performance and memory bandwidth at different cache levels to guide optimization.

- **License**: Apache-2.0
- **Language**: Python ≥3.9 (primary), C (measurement harness), inline assembly (ISA-specific benchmarks)
- **Outputs**: CSV, JSON, table, plots in platform user data dir for `carm`, or `--output-file`
- **Entry point**: `carm` console script → `carm_roofline.carm:main`
- **Install**: `pip install -e .` or `pip install -e ".[all]"`
- **C compiler**: gcc ≥4.9

## Commands

The `carm` CLI has three subcommands:

| Command | Purpose |
|---------|---------|
| `carm benchmark` | Run benchmarks (arithmetic, memory, roofline) to construct performance models |
| `carm profile` | Profile instrumented applications (MPI, threaded, hybrid) for roofline metrics |
| `carm gui` | Launch the interactive Dash+Plotly roofline dashboard |

Run `carm <command> --help` for command-specific options. Configuration classes inject arguments modularly via `InsertsArguments`.

## Architecture & Data Flow

### Pipeline (six stages)

```
CLI args → CARMContext → generate → compile → execute → parse → output
```

1. **Generate**: For each ISA, `generate_microbenchmarks()` calls `ISABenchmarkSuite.generate()` which produces inline assembly `MicrobenchmarkFunctionSpec` objects
2. **Flatten**: All per-ISA benchmarks merged into a single `dict[str, BaseBenchmark]`
3. **Header**: `create_microbenchmark_header()` renders C header with inline benchmark functions, metadata structs, and wrapper instantiations via X-macro pattern
4. **Compile**: `compile_test_bench()` builds a binary via `ExecutionInterface.compile()`
5. **Execute**: `run_microbenchmarks()` runs with physics-based timeout, returns CSV-formatted stdout
6. **Parse**: `parse_benchmark_output()` populates benchmark result fields

## Repository Layout

```
carm_roofline/            Main Python package
├── architecture/         Hardware detection (CPU, cache, ISA features, frequency)
│   ├── architecture.py   Architecture class, ISAFrequencies
│   ├── detect.py         DetectedArchitecture, native_detect/detect_for_isa
│   ├── memory.py         MemoryTopology (sysfs), SimpleMemoryTopology (CLI/TOML)
│   └── tests/            C probe source files per ISA family
├── benchmark/            Benchmark system
│   ├── generation/       ISA-specific code generation (instructions, registers, parameters)
│   ├── suites/           Benchmark suites (arithmetic, memory, roofline, sweep)
│   ├── output/           Strategy-pattern output dispatch (table/plot/json/csv)
│   ├── interface.py      run_full_benchmark() orchestration
│   └── benchmarking.py   Benchmarking config (TestType, CLI args)
├── core/                 Domain primitives (DataType, Operation, UserError, type-safe units)
├── isa/                  ISA identity hierarchy (BaseISA, ALL_ISAS, ISA_NAME_TO_CLASS)
├── test_bench/           C measurement harness (builder.py, test_bench.c/h, wrapper.inl)
├── profiling/            PAPI/perf application profiling pipeline
├── gui/                  Dash+Plotly interactive roofline dashboard
├── carm.py               Entry point, CLI parser, subcommand dispatch
├── context.py            CARMContext dataclass (architecture, benchmarking, exec_interface, run_config)
├── arguments.py          InsertsArguments, validators, enum_action, TopLevelHelpFormatter
├── exec_interface.py     Compile/run abstraction with simulator support
├── run_config.py         RunConfig (verbose, dry-run, output format)
├── output_utils.py       Verbosity-leveled Rich console output
├── workspace.py          workspace_context() temp directory manager
└── results_paths.py      default_results_root() via platformdirs
test/                     Pytest test suite (unit + integration)
docs/                     GitHub Pages website source (Jekyll): user docs, quickstart, command reference
examples/                 Usage examples (e.g., LULESH profiling)
pyproject.toml            Single source of truth: dependencies, ruff, mypy, pytest config
.pre-commit-config.yaml   Pre-commit hooks (ruff, mypy, clang-format, trailing-whitespace)
```

## Development Commands

```bash
# Run benchmarks (auto-detects all ISAs)
carm benchmark --test arithmetic --test-time 1

# Specific ISA, memory test
carm benchmark --isa x86_avx2 --test memory --mem-target L1

# Dry run (generate code only)
carm benchmark --test arithmetic --dry-run --verbose 4

# Tests
pytest -m unit test/ # Fast unit tests
pytest # All tests

# Lint, format, type-check
ruff check --fix
ruff format
mypy .
pre-commit run --all-files
```

## Code Conventions & Common Patterns

### Formatting & Style

- Ruff-managed: line-length 120, quote-style double, indent-style space, LF endings
- Target Python 3.9
- `from __future__ import annotations` in every file (`|` union types are allowed)
- Ruff and mypy exclude `test`
- C code: clang-format with style=file (`.clang-format` at root)

### Naming

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private module internals: `_leading_underscore`
- Type variables: short single uppercase (e.g. `T` for `Unit[T]`)

### Modular Argument Injection

All configuration classes inherit from `InsertsArguments` and define `insert_arguments()`:

```python
class MyConfig(InsertsArguments):
    @staticmethod
    def insert_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--my-arg", type=str, default="value")

    def __init__(self, args: argparse.Namespace):
        self.my_arg = args.my_arg
```

Classes using this pattern: `Architecture`, `Benchmarking`, `RunConfig`, `ExecutionInterface`, `ProfileConfig`, `GUIConfig`. Discovery is automatic via `InsertsArguments.subclasses()` BFS.

### Context Variable Pattern

`ExecutionInterface` is shared via `contextvars` to avoid parameter threading through deep detection code:

```python
from carm_roofline.architecture import set_execution_interface, get_execution_interface
exec_iface = get_execution_interface()
exec_iface.compile(...)
```

### ISA Registration

ISAs register via explicit tuples in `carm_roofline/isa/__init__.py`:

```python
ALL_ISAS = (ArmScalar, ArmNeon, ArmSVE, RISCVScalar, RISCV_RVV_071, RISCV_RVV, X86Scalar, ...)
ISA_NAME_TO_CLASS = {"x86_avx2": X86AVX2, ...}
INCOMPATIBLE_ISAS = {frozenset({RISCV_RVV_071, RISCV_RVV})}
```

### Error Handling

- `UserError(Exception)` for expected user misconfiguration — caught in `carm_roofline.carm:main()`, exits 1 without traceback
- `ValueError` for bad CLI args (exit 1)
- Generic exceptions exit 2
- No silent failures — prefer exceptions over fallback values

### Output Dispatch

Strategy pattern via `OutputHandler` protocol (in `benchmark/output/`). Dispatch map keyed by `TestType` enum, built in `benchmark/output/__init__.py`.

### Type-Safe Units (`core/units.py`)

Generic arithmetic wrapper `Unit[T]` (ABC) with subclasses: `Bytes`, `Operations`, `Frequency`, `Bandwidth`, `Performance`, `Seconds`, `Cycles`, `ArithmeticIntensity`. Automatic prefix selection (`str(b) → "8.00 MiB"`). Factory methods like `Performance.from_ops_per_second()`.

## Testing & QA

- **Framework**: pytest ≥7.0, configured in `pyproject.toml`
- **Root**: `test/`, with `test/unit/` for fast unit tests
- **Markers** (7 defined): `unit`, `integration`, `slow`, `golden`, `x86`, `arm`, `riscv` — only `unit` is actively used
- **Fixtures**: 5 in `test/conftest.py` — `x86avx_isa`, `x86sse_isa`, `arm_neon_isa`, `riscv_scalar_isa`, `mock_context`
- **Mocking**: lightweight — `unittest.mock.Mock` for context, `monkeypatch` for module-level intercepts
- **Dominant pattern**: parametrized ISA cross-product tests via `@pytest.mark.parametrize`
- **Categories**: Unit (`test/unit/`, marked `unit`) for profiling model, CLI smoke, register abstraction, ISA helpers. Integration-ish (`test/` root, no markers) for ISA codegen (11-ISA matrix), typed benchmarks, output handlers, memory suite generation.
