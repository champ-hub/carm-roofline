# Repository Guidelines

## Project Overview

CARM (Cache-Aware Roofline Model) is a micro-benchmarking toolkit that constructs roofline performance models across multiple CPU architectures (x86, ARM, RISC-V) and GPU platforms (ROCm, CUDA). It measures arithmetic performance and memory bandwidth at different cache levels to guide optimization.

- **License**: LGPL-2.1
- **Language**: Python ≥3.9 (primary), C (measurement harness), inline assembly (ISA-specific benchmarks)
- **Outputs**: CSV, JSON, table, plots in platform user data dir for `carm`, or `--output-file`

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

### Key Modules

| Module | Role |
|--------|------|
| `carm.py` | Main entry point, CLI parser, subcommand dispatch |
| `context.py` | `CARMContext` dataclass threading through pipeline |
| `architecture/` | Hardware auto-detection (CPU, cache, ISA features, frequency) |
| `benchmark/generation/` | ISA-specific inline assembly code generation |
| `benchmark/suites/` | Benchmark suite orchestration (arithmetic, memory, roofline) |
| `benchmark/output/` | Strategy-pattern output dispatch (table/plot/json/csv) |
| `test_bench/` | C measurement harness (calibration, timing, threading) |
| `profiling/` | PAPI/perf application profiling pipeline |
| `gui/` | Dash+Plotly interactive roofline dashboard |
| `arguments.py` | `InsertsArguments` base class for modular argument injection |
| `exec_interface.py` | Native/simulated/cross-compiled command execution |
| `run_config.py` | Run configuration (verbosity, output format, dry-run) |
| `units.py` | Type-safe unit wrappers (Bytes, Frequency, Performance, Bandwidth, etc.) |
| `docs/` | GitHub Pages website (Jekyll): user-facing docs, quickstart, command reference |

## Key Directories

```
├── architecture/         Hardware detection (C probes, sysfs parsing, ISA feature discovery)
│   ├── architecture.py   Architecture class, ISAFrequencies
│   ├── detect.py         DetectedArchitecture, DetectionBuilder, native_detect/detect_for_isa
│   ├── memory.py         MemoryTopology (sysfs), SimpleMemoryTopology (CLI/TOML)
│   └── tests/            C probe source files per ISA family
├── benchmark/            Benchmark system facade
│   ├── generation/       ISA code generation (BaseISA hierarchy, code_gen utilities)
│   ├── suites/           Benchmark suite classes (arithmetic, memory, roofline, sweep)
│   ├── output/           Strategy-pattern output handlers per test type
│   ├── interface.py      run_full_benchmark() orchestration
│   └── benchmarking.py   Benchmarking config (TestType, CLI args)
├── test_bench/           C measurement harness (builder.py, test_bench.c/h, wrapper.inl)
├── profiling/            Application profiling pipeline (PAPI/perf backends)
├── gui/                  Dash+Plotly interactive dashboard
├── test/                 Pytest test suite (unit/integration)
├── docs/                 GitHub Pages website source (Jekyll): user docs, quickstart
└── carm.py               Main entry point
```

## Development Commands

```bash
# Run benchmarks (auto-detect ISA)
./carm.py benchmark --test arithmetic --num-ops 1000 --test-time 1

# Specific ISA, memory test, CSV output
./carm.py benchmark --isa x86_avx2 --test memory --mem-target L1 --output-format csv

# Cross-compile for RISC-V with QEMU
./carm.py benchmark --isa riscv_rvv --compiler riscv64-linux-gnu-gcc \
    --sim-cmd "qemu-riscv64 {binary}" --test roofline --output-format json

# Dry run (generate code only)
./carm.py benchmark --test arithmetic --dry-run --verbose 4

# Unit tests (fast)
pytest -m unit test/

# All tests
pytest -v

# Lint & format
ruff check --fix .
ruff format .

# Type check
mypy .

# Full pre-commit
pre-commit run --all-files
```

## Code Conventions & Common Patterns

### Formatting & Style

- Ruff-managed: line-length 120, quote-style double, indent-style space, LF endings
- Target Python 3.9 (no 3.10+ match statements in production code)
- `from __future__ import annotations` in every file
- Excluded from ruff/mypy: `legacy_bench_gen/`, `run.py`, `run_gpu.py`, `*AI_Calculator.py`, `utils.py`, `output_utils.py`, `gui/dashboard.py`, `gui/gui_utils.py`, `test/`
- C code: clang-format with style=file (`.clang-format` at root)

### Naming

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private module internals: `_leading_underscore`
- Type variables: short single uppercase (e.g. `T` for `Unit[T]`)

### Modular Argument Injection Pattern

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
from architecture import set_execution_interface, get_execution_interface
exec_iface = get_execution_interface()
exec_iface.compile(...)
```

### ISA Registration

ISAs register via explicit tuples in `benchmark/generation/__init__.py`:

```python
ALL_ISAS = (X86AVX512, X86AVX2, X86AVX, X86SSE, X86Scalar, ...)
ISA_NAME_TO_CLASS = {"x86_avx512": X86AVX512, ...}
INCOMPATIBLE_ISAS: set[frozenset[type[BaseISA]]] = {frozenset({X86AVX, X86AVX2})}
```

### Error Handling

- `UserError(Exception)` for expected user misconfiguration — caught in `carm.py main()`, exits 1 without traceback
- `ValueError` for bad CLI args (exit 1)
- Generic exceptions exit 2
- No silent failures — prefer exceptions over fallback values

### Benchmark Pipeline Flow

```python
# In benchmark/interface.py run_full_benchmark():
# 1. Generate suites per ISA
suites = {isa: generate_microbenchmarks(context, isa) for isa in context.architecture.isa_names}
# 2. Flatten all benchmarks
flat = {name: b for s in suites.values() for name, b in s.benchmarks.items()}
# 3. Create header → compile → run → parse
create_microbenchmark_header(flat.values(), header_path)
compile_test_bench(context, binary_path, include_dirs)
output = run_microbenchmarks(context, binary_path, flat.values())
parse_benchmark_output(flat, output)
```

### Output Dispatch

Strategy pattern via `OutputHandler` protocol (in `benchmark/output/`):

```python
class ArithmeticOutputHandler:
    @staticmethod
    def handle(context, isa_suites) -> None:
        # Dispatches to print_table()/write_plot()/write_csv()/write_json()
        # based on context.run_config.output_formats
```

Dispatch map keyed by `TestType` enum, built in `benchmark/output/__init__.py`.

### Type-Safe Units (`units.py`)

```python
class Unit[T](ABC):  # Generic arithmetic wrapper
# Subclasses: Bytes, Operations, Frequency, Bandwidth, Performance, Seconds, Cycles, ArithmeticIntensity
# Automatic prefix selection: str(b) → "8.00 MiB"
# Factory methods: Performance.from_ops_per_second()
```

## Important Files

| File | Purpose |
|------|---------|
| `carm.py` | Entry point, CLI parser, subcommand dispatch |
| `context.py` | `CARMContext` dataclass (architecture, benchmarking, exec_interface, run_config) |
| `arguments.py` | `InsertsArguments`, validators, `enum_action`, `TopLevelHelpFormatter` |
| `error.py` | `UserError` exception class |
| `exec_interface.py` | Compile/run abstraction with simulator support |
| `run_config.py` | `RunConfig` with verbose/dry-run/output format settings |
| `output_utils.py` | Verbosity-leveled Rich console output |
| `units.py` | Type-safe unit wrappers |
| `workspace.py` | `workspace_context()` temp directory manager |
| `results_paths.py` | `default_results_root()` via platformdirs |
| `pyproject.toml` | Single source of truth: dependencies, ruff, mypy, pytest config |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, mypy, clang-format, trailing-whitespace) |
| `docs/` | GitHub Pages website source (Jekyll): user docs, quickstart, command reference |
| `.github/copilot-instructions.md` | (Loaded as system context) Central developer guidance, module index |
| `test/conftest.py` | Shared pytest fixtures (ISA instances, mock context) |


## Runtime/Tooling Preferences

- **Python**: ≥3.9, setuptools ≥64.0 + wheel build system
- **Package manager**: pip (no poetry, no conda)
- **Entry point**: `carm` console script → `carm:main`
- **C compiler**: gcc ≥4.9 (AVX-512 requires ≥9.3)
- **Optional deps**: [gui] Dash+Plotly, [dev] pytest, ruff, mypy, pre-commit
- **Install**: `pip install -e .` or `pip install -e ".[dev,gui]"`
- **VS Code**: `.vscode/settings.json`, `launch.json`, `c_cpp_properties.json` provided

## Testing & QA

- **Framework**: pytest ≥7.0, configured in `pyproject.toml [tool.pytest.ini_options]`
- **Root**: `test/` (testpaths), `test/unit/` for fast unit tests
- **Markers** (6 defined, only `unit` actively used): `unit`, `integration`, `slow`, `golden`, `x86`, `arm`, `riscv`
- **Fixtures**: 5 in `test/conftest.py` — `x86avx_isa`, `x86sse_isa`, `arm_neon_isa`, `riscv_scalar_isa`, `mock_context`
- **Mocking**: lightweight — `unittest.mock.Mock` for context, `monkeypatch` for module-level intercepts
- **Dominant pattern**: parametrized ISA cross-product tests via `@pytest.mark.parametrize`
- **Test categories**:
  - Unit (`test/unit/`, marked `@pytest.mark.unit`): profiling model, CLI smoke, register abstraction, ISA helpers
  - Integration-ish (`test/` root, no markers): ISA codegen (11-ISA matrix), typed benchmarks, output handlers, memory suite generation
- **Coverage**: 11 ISA classes tested across arithmetic/memory codegen, edge cases (single op, boundaries, thread scaling, underflow)
- **Deprecated**: `refactor_tests/` directory no longer exists — all tests migrated to `test/`
