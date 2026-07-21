---
title: Profiling
parent: Commands
---

# `carm profile`

Profile instrumented applications to compute roofline metrics: arithmetic intensity, FLOP/s, and memory bandwidth. Supports MPI, threaded, and hybrid (MPI + OpenMP) applications.

Run `carm profile --help` for a full argument listing.

## Basic Usage

```bash
carm profile [options] -- <command>
```

Everything after `--` is the application command to profile, passed verbatim.

### Examples

```bash
# Profile a simple binary with default PAPI counters
carm profile -- ./my_app --my --args

# Profile an MPI application with perf backend
carm profile --backend perf -- mpirun -np 4 ./my_mpi_app

# Custom PAPI events with cross-thread region-based aggregation
carm profile --backend papi --aggregation region_merged -- ./my_app

# Specify one or more ISA vector widths for custom PAPI FLOPS/BYTES metrics
carm profile --isa x86_avx2 --data-type f64 -- ./my_app

# Multiple ISAs — union of their FP_ARITH counters avoids counter budget overflow
carm profile --isa x86_avx2 x86_sse x86_scalar --data-type f64 -- ./my_app
```

## Arguments by Category

### Positional: the profiled command

The `command` positional argument (everything after `--`) is the application to profile. It is passed to the runner as-is, preserving flags and arguments.

### Profiler backend (`--backend`)

Choose between PAPI (requires PAPI development libraries) and `perf` (uses Linux's `perf`, no external library). The PAPI backend generally provides more accurate counter readings; `perf` is a lighter alternative when PAPI isn't available.

### Sampling mode (`--perf-interval`)

By default `perf` runs in full-execution mode, accumulating counters over the entire run. Pass `--perf-interval` to enable interval-based sampling, producing a time series of counter values at the specified granularity (in milliseconds). Doesn't apply to PAPI.

### Aggregation (`--aggregation`)

Controls how multi-rank (MPI) or multi-thread results are combined:

| Mode | What it does |
|------|--------------|
| `region_merged` | One point per region. Instances of the same region across different ranks/threads are merged. **Best option for a properly-annotated application**.|
| `global` | One point for the whole application. Merges all ranks and threads. |
| `rank` | One point per rank, merges threads. |
| `thread` | One point per thread, merges regions. |
| `region_per_thread` | One point per region per thread. |

### Hardware assumptions (`--isa`, `--data-type`)

When hardware counters can't directly count FLOPs (e.g. on older Intel or AMD CPUs), the tool estimates operations from instruction counts. `--isa` and `--data-type` tell it how many FLOPs each instruction retired, so the estimate is more accurate for your code's actual vector ISA and precision.

In Intel processors, specifying different `--isa` values (e.g. `--isa x86_avx2 x86_scalar`) allows the CARM Tool to use a minimal set of FP_ARITH counters, targeting only those ISAs. This help avoid exceeding the hardware counter budget, which leads to incorrect results. If you get a warning about the resolved events not fitting the available hardware counters, try specifying fewer ISAs, omitting those your application doesn't use.

### Output and naming (`--verbose`, `--machine-name`, `--app-name`, `--output-dir`, `--keep-artifacts`)

Results go to `--output-dir` (default: platform user data dir) under a machine-specific subdirectory. `--machine-name` overrides the auto-detected CPU model name (note: you should use the same name as `carm benchmark --name`); `--app-name` overrides the name extracted from the command. `--keep-artifacts` preserves raw profiler output files for debugging. `--verbose` increases detail.

## Backend Selection Details

| | PAPI | Perf |
|---|---|---|
| Dependency | PAPI dev libraries (`libpapi`) | None (kernel built-in) |
| Sampling | Various aggregation modes | Full-execution or interval (`--perf-interval`) |
| Insight | Provides per-region metrics if the application is annotated | Characterizes the application across time, but can't distinguish code regions |

## Output

Profiling results are written to the output directory. Key outputs:

- **CSV/JSONL files** with per-region metrics (arithmetic intensity, FLOP/s, bandwidth)
- **Roofline-compatible data** for `carm gui` interactive exploration
