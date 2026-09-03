---
title: Profiling
parent: Commands
nav_order: 3
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

# Custom aggregation for region-based analysis
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

### Optional metrics (`--metrics`, `--list-metrics`)

The CLI is metric-centric: FLOPS and BYTES are always collected (needed for roofline plotting). Other metrics are optional and selected by name:

```bash
# List the available optional metrics
carm profile --list-metrics

# Profile with the cache-residency metric
carm profile --metrics cache-residency -- ./my_app

# Merge runs when the required events exceed the hardware counter budget
carm profile --merge-runs --metrics cache-residency -- ./my_app
```

Each optional metric maps to a set of hardware events chosen from what your system supports. A profile always runs the command once by default; over-budget events are dropped and a warning lists them. `--merge-runs` will run the command multiple times to collect all requested metrics, merging the results from multiple runs.

The `cache-residency` metric reports, per region, the fraction of memory traffic served at each cache level, with per-level resident bytes. On AMD, the L1 miss rate is scaled to using the declared data-type/ISA bytes-per-instruction, so pass `--data-type` matching the workload (e.g. `--data-type f64` for f64 code) and `--isa` when vectorized. The AMD event set can exceed the hardware counter budget, so use `--merge-runs` to collect it across multiple runs.

The `cache-line-utilization` metric reports `CLU = application_bytes / (L1_data_misses * 64)`. It uses a fixed 64-byte cache-line assumption. CLU can exceed 100% because repeated accesses count in application bytes. Note that this is a core-centric approximation of the actual cache-line utilization, and will report inaccurate results in certain cases (e.g. the application repeatedly reads and writes the same 4-8 bytes of each cache line)

Build and profile the example:

```bash
make -C examples/cache-line-utilization
carm profile --backend papi --aggregation global --data-type f32 --isa x86 \
  --metrics cache-line-utilization --merge-runs -- \
  ./examples/cache-line-utilization/cache_line_utilization 32 200 100
```

### Output and naming (`--verbose`, `--machine-name`, `--app-name`, `--output-dir`, `--keep-artifacts`)

Results go to `--output-dir` (default: platform user data dir) under a machine-specific subdirectory. `--machine-name` overrides the auto-detected CPU model name (note: you should use the same name as `carm benchmark --name`); `--app-name` overrides the name extracted from the command. `--keep-artifacts` preserves raw profiler output files for debugging. `--verbose` increases detail.

## Instrumenting Your Application

The PAPI backend captures per-region hardware-counter data by wrapping code regions with `PAPI_hl_region_begin` / `PAPI_hl_region_end`.  How you place these calls determines whether CARM sees the work done by **every thread** or only the master thread.

### Canonical example

The repository includes a [PAPI-instrumented fork of LULESH 2.0](https://github.com/LLNL/LULESH) under `examples/lulesh-papi/`.  Study its instrumentation pattern before annotating your own code.  Key annotations in that example:

| Region | Instrumentation scope |
|--------|----------------------|
| `CalcKinematics`, `CalcMonotonicQGradients`, `CalcMonotonicQRegion`, `EvalEOS_setup`, `UpdateVolumes` | **Per-thread** — inside `#pragma omp parallel` |
| `LagrangeNodal` | **Serial wrapper** — master thread only (noted explicitly in source) |

### Required initialisation

At process startup, before any region annotation:

1. Call `PAPI_library_init(PAPI_VER_CURRENT)` once.
2. For OpenMP applications, call `PAPI_thread_init` so PAPI can distinguish threads:

```c
PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num);
```

Without `PAPI_thread_init`, PAPI cannot associate counters with individual OpenMP threads, and multi-threaded results will be incorrect or missing.

### Correct pattern: per-thread region annotation

Place `PAPI_hl_region_begin` and `PAPI_hl_region_end` **inside** the `#pragma omp parallel` block so that every thread enters and exits the region:

```c
#pragma omp parallel
{
    PAPI_hl_region_begin("MyKernel");
    #pragma omp for
    for (int i = 0; i < N; ++i) {
        /* kernel work */
    }
    PAPI_hl_region_end("MyKernel");
}
```

Each thread accumulates its own hardware counters for the region.  CARM's `--aggregation` modes (e.g., `region_merged`, `thread`, `region_per_thread`) can then correctly combine or separate per-thread measurements.

### Wrong pattern: wrapping outside the parallel region

```c
/* WRONG — only the master thread calls PAPI_hl_region_begin/end */
PAPI_hl_region_begin("MyKernel");
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    /* kernel work */
}
PAPI_hl_region_end("MyKernel");
```

**What happens**: Only the thread that called `PAPI_hl_region_begin` (the master thread) records counter data. Every other OpenMP thread executes the loop work but *never enters the PAPI region*. Their hardware-counter contributions are invisible to PAPI.

### Checking your instrumentation

Run a quick sanity check with `--aggregation thread`:

```bash
carm profile --backend papi --aggregation thread -- ./my_app
```

If you annotated correctly, you should see one line per thread. If only one thread appears for a parallel kernel, your region calls are outside the parallel block.

## Backend Selection Details

| | PAPI | Perf |
|---|---|---|
| Dependency | PAPI dev libraries (`libpapi`) | None (kernel built-in) |
| Sampling | Various aggregation modes | Full-execution or interval (`--perf-interval`) |
| Insight | Provides per-region metrics if the application is annotated | Characterizes the application across time, but can't distinguish code regions |

## Output

Profiling results are written to `<output_dir>/<machine_name>/` (the machine name is auto-detected or set with `--machine-name`). Key outputs:

- **`applications.jsonl`** — one appended JSON line per run, embedding run metadata and the aggregated roofline points. **This is the file `carm gui` reads.**
- **`machine.json`** — machine-signature debug file written on the first run.
- **Console summary** — one line per aggregated point: `label: AI=… FLOP/Byte, … GFLOP/s, … GB/s, …s`
