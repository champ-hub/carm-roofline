---
title: Benchmarking
parent: Commands
---

# `carm benchmark`

Run micro-benchmarks to measure peak arithmetic performance and memory bandwidth across cache levels. This is the primary subcommand for constructing the Cache-Aware Roofline Model.

Run `carm benchmark --help` for a complete listing of all arguments, types, and defaults.

## Basic Usage

```bash
# Full roofline benchmark (all available ISAs)
carm benchmark --test roofline

# Thorough exploration of the parameter space
# Automatically tests all parameter combinations
carm benchmark --test roofline \
    --isa x86 x86_avx2 x86_avx512 \
    --data-type f32 f64 \
    --ld-st-ratio 2:1 1:1 1:0 0:1 \
    --threads 1 4 8 16 \
    --instruction add fma div

# Arithmetic-only test with specific data types
carm benchmark --test arithmetic --data-type f32 f64 --instruction add fma

# Memory bandwidth test targeting L1 only
carm benchmark --test memory --mem-target L1 --ld-st-ratio 2:1

# Dry run (generate code only, inspect generated assembly)
carm benchmark --test arithmetic --dry-run

# Generate a TOML config template for custom cache topology
carm benchmark --emit-config my_topology.toml
```

## Test Types

| Test | Use this to... |
|------|----------------|
| `roofline` | **Default.** Get the full picture: peak compute + bandwidth for each cache level in one run, constructing the CARM. |
| `arithmetic` | Measure GFLOP/s for specific instruction/data-type combinations. Useful when you only care about peak performance. |
| `memory` | Measure bandwidth for specific cache levels. Useful when you only care about memory bandwidth. |
| `memory_sweep` | Profile bandwidth across a range of buffer sizes to observe the full memory bandwidth curve. |

## Arguments by Category

### Which ISAs and hardware to target (`--isa`, `--vector-length`, `--vector-lmul`, `--frequency`)

By default the tool auto-detects all available ISAs on your system. Use `--isa` to restrict testing to a specific ISA (e.g. `x86_avx2`), which is useful when cross-compiling or when you only care about one vector extension. `--vector-length` overrides the detected vector register length for RVV/SVE. `--vector-lmul` sets the RISC-V LMUL group multiplier. `--frequency` overrides the detected frequency, make sure you use the correct value to avoid incorrect results. `--set-frequency` applies the aforementioned frequency (x86 only, requires root).

### Custom cache topology (`--topology-config`, `--emit-config`)

The tool auto-detects your cache hierarchy from sysfs. If that fails or you want to model a different system, use `--emit-config` to dump a template TOML file, edit it with your cache parameters, and pass it via `--topology-config`.

### What to measure (`--test`, `--data-type`, `--instruction`, `--mem-target`, `--ld-st-ratio`, `--arith-mem-ratio`, `--mem-test-sizes`, `--num-ops`)

These control the test performed on the system. The main selection is `--test`, described above. Within a test you can narrow further: restrict data types (`--data-type`), pick arithmetic instructions (`--instruction`), choose which memory levels to target (`--mem-target`), and control the load-to-store ratio (`--ld-st-ratio`). For the `mixed` test, `--arith-mem-ratio` controls how many arithmetic operations per memory access. `--mem-test-sizes` lets you set exact array sizes per cache level instead of auto-sizing. `--num-ops` sets how many arithmetic operations the inner loop performs.

### How long to run (`--test-time`)

Each micro-benchmark loops until `--test-time` seconds elapse. The default (25s) gives reliable results on most systems. You can reduce this for a quicker benchmark, but accuracy may suffer. However, even 1 second is enough for a rough comparison between machines/configurations.

### Threading (`--threads`, `--interleaved`)

Benchmark with multiple threads by passing one or more thread counts, e.g. `--threads 1 4 8`. The tool will run the full suite at each count. `--interleaved` changes thread pinning for NUMA systems where core indices are interleaved across nodes (e.g. node 0 has cores 0, 2, 4…).

### Output and verbosity (`--verbose`, `--name`, `--dry-run`, `--output-dir`, `--output-fmt`, `--keep-artifacts`)

Results land in a directory under `--output-dir` (default: platform user data dir). `--output-fmt` controls which formats are written (table, plot, csv, jsonl). `--name` sets a machine name instead of the auto-generated one (derived from CPU model). `--dry-run` skips compilation and execution, useful for inspecting generated assembly. `--keep-artifacts` preserves the temporary source/binaries for debugging. Use `--verbose/-v [0-4]` for more detail (`-v 4` for full debug).

### Cross-compilation and simulation (`--compiler`, `--sim-cmd`)

**Under development**

Target a non-native architecture by setting `--compiler` to a cross-compiler (e.g. `riscv64-linux-gnu-gcc`) and optionally `--sim-cmd` to run the binary under a simulator. Use `{binary}` as a placeholder in the sim command, e.g. `--sim-cmd "qemu-riscv64 {binary}"` or `--sim-cmd "sde -mix -- {binary}"`.

## Output Files

Results are written to the output directory in a subdirectory named after the run. Typical outputs:

- **Roofline plot** (PNG): visual roofline model with measured points
- **Arithmetic performance plot**: GFLOP/s across operations and data types
- **Memory bandwidth plot**: bandwidth per cache level
- **CSV/JSONL**: raw data consumed by `carm gui` for roofline visualization
