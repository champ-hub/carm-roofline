# Legacy Components Documentation Index

This directory documents CARM's legacy/deprecated components for reference and migration purposes.
All new development should use the refactored modules described in [copilot-instructions.md](../copilot-instructions.md).

## Quick Lookup

| I need to understand...                          | Read                                             |
|--------------------------------------------------|--------------------------------------------------|
| Old CLI arguments from `run.py`                  | [run-py.md](run-py.md)                           |
| Old vs new benchmark generation pipeline         | [run-py.md § Pipeline](run-py.md)                |
| C-based assembly code generator (`legacy_bench_gen/`) | [bench-gen.md](bench-gen.md)               |
| `Bench/` directory and its Makefile build system | [bench-gen.md § Bench Dir](bench-gen.md)         |
| GPU roofline measurement (ROCm/CUDA)             | [gpu-and-calculators.md](gpu-and-calculators.md) |
| Measuring application arithmetic intensity       | [gpu-and-calculators.md § Calculators](gpu-and-calculators.md) |
| PAPI / DynamoRIO / SDE / rocprofv3 / Nsight backends | [gpu-and-calculators.md § AI Calculators](gpu-and-calculators.md) |
| `gui/dashboard.py` web dashboard                 | [utility-scripts.md § Results Visualization](utility-scripts.md) |
| Memory bandwidth plotting scripts                | [utility-scripts.md § Standalone Scripts](utility-scripts.md) |
| Shared utilities (`utils.py`, `output_utils.py`, `units.py`) | [utility-scripts.md § Shared Modules](utility-scripts.md) |

---

## Document Summaries

### [run-py.md](run-py.md) — Legacy `run.py` CLI Entry Point

The deprecated `run.py` was a monolithic ~1440-line driver that:
- Accepted a rich set of CLI arguments including multi-precision sweeps, multi-thread sweeps, ISA pinning, and hardcoded CSV output
- Drove a two-binary pipeline (`Bench/Bench` for config, `bin/test` for timing) calling external Make targets
- Computed GB/s and GFLOP/s manually per architecture in Python
- Wrote a hardcoded two-row-header CSV schema

Replaced by: `carm.py` + the full refactored architecture. Documents all old CLI flags, the old execution flow, and a diff-table of what changed.

---

### [bench-gen.md](bench-gen.md) — Legacy C-Based Benchmark Generator

The deprecated `legacy_bench_gen/` (and `Bench/`) used a C-based two-phase approach:
- ISA-specific generator binaries (one per ISA, selected via preprocessor defines like `-DAVX2`)
- Wrote inline assembly into `Test/test_params.h` via `fprintf` in `write_asm.c`
- Shelled out to `make` to compile the final benchmark
- All ISA config encoded as string macros in `config_test.h`
- Cache/memory parameters calculated by `calc_param.c`

Replaced by: `benchmark/generation/` Python system with in-memory generation and typed class hierarchies. Documents all C source files, configuration macros, ISA selection mechanism, and migration notes.

---

### [gpu-and-calculators.md](gpu-and-calculators.md) — GPU Benchmarking & AI Calculators

Two distinct legacy/specialized subsystems:

**`run_gpu.py`** — Self-contained GPU roofline ceiling measurer:
- Auto-detects NVIDIA/AMD hardware; builds CUDA/HIP micro-benchmarks via `GPU/Bench/`
- Iterates all supported vector/tensor precisions, collects FLOP/s and memory bandwidth (shared, L2, global)
- Outputs per-precision CSV rows; still the only GPU measurement path in the project

**`*_AI_Calculator.py` scripts** — Application profilers that place workloads on the roofline chart:
- `PMU_AI_Calculator.py` — PAPI hardware performance counters
- `DBI_AI_Calculator.py` — DynamoRIO binary instrumentation
- `SDE_AI_Calculator.py` — Intel SDE emulation
- `ROC_AI_Calculator.py` — rocprofv3 (AMD)
- `NCU_AI_Calculator.py` — NVIDIA Nsight Compute

All produce the same `AI/Gflops/Bandwidth` CSV columns via shared output utilities.

---

### [utility-scripts.md](utility-scripts.md) — Utility & Analysis Scripts

Standalone tools and shared modules:

**Standalone analysis/debug scripts:**
- `calc_stream.py` — Theoretical STREAM bandwidth calculator
- `plot_memory_bandwidth.py` — Plots bandwidth curves from result files
- `plot_timing_distribution.py` — Plots timing measurement distributions
- `array_pattern_cmp.c` — Low-level memory access pattern comparison

**Shared utility modules** (used by both legacy and refactored code):
- `utils.py` — Legacy matplotlib plotting + CSV I/O, Rich-based verbosity control
- `output_utils.py` — Shared CSV/table output formatting
- `units.py` — Type-safe physical unit definitions and conversions

**Visualization:**
- `gui/dashboard.py` — 3614-line Dash web dashboard; reads `carm_results/roofline/*.csv` and `carm_results/applications/*.csv`; 50+ reactive callbacks for interactive roofline charts and cross-machine comparison
- `gui/gui_utils.py` — CSV parsing, roofline geometry, and Plotly trace generation helpers for the GUI
