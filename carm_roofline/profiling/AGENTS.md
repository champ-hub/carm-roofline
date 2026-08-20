# Profile Module — MPI-Aware Application Profiling

The `profile` package provides a pipeline for profiling instrumented applications (PAPI HL or Linux perf), computing roofline metrics (arithmetic intensity, FLOP/s, bandwidth) with first-class support for MPI ranks and threads.

## Architecture

### Module Overview

| Module | Contents |
|---|---|
| `backends.py` | `ProfilerBackend` ABC + `create_backend()` factory |
| `papi_backend.py` | `PAPIHLBackend` — PAPI High-Level API backend |
| `perf_backend.py` | `PerfBackend` — Linux perf stat backend |
| `shared.py` | Backend-agnostic types: `MetricType`, `MetricDefinition`, `MetricResolutionConfig`, `MetricContext`, `compute_region_point()`, `sum_roofline_points()`, `resolve_metrics()`, `check_perf_event_paranoid()` |
| `papi_metrics.py` | PAPI-specific metric definitions, event discovery, `resolve_metrics()` wrapper |
| `perf_metrics.py` | Perf-specific metric definitions, event discovery, `resolve_perf_metrics()` |
| `papi_loader.py` | PAPI HL rank-file discovery and parsing |
| `perf_loader.py` | Perf CSV output parsing |
| `papi_lib.py` | ctypes access to libpapi: library discovery/loading and the per-event collectability test |
| `aggregation.py` | Aggregation strategies (`global`, `per-rank`, `per-thread`, `region_merged`, `region_per_thread`) |
| `merge.py` | Run partitioning (`partition_events`) and merging (`merge_runs`, `missing_required_events`) |
| `optional_metrics.py` | Optional metrics registry (e.g. `cache-residency`), validation and resolution |
| `config.py` | `ProfileConfig` + argument parsing |
| `model.py` | Data model: `RegionMetrics`, `ThreadMetrics`, `RankMetrics`, `RunMetadata`, `RunResults` |
| `output.py` | Output writers (JSONL, machine JSON) |

### Data Model (`model.py`)

```
Run (metadata)
├── Rank 0
│   ├── Thread 0 (regions with raw counters)
│   ├── Thread 1
│   └── ...
├── Rank 1
│   ├── Thread 0
│   └── ...
└── ...
```

- `RegionMetrics`: Raw performance counters for one annotated code region.
- `ThreadMetrics`: Collection of regions for a thread within an MPI rank.
- `RankMetrics`: Collection of threads for an MPI rank.
- `RunResults`: Complete profiling result with metadata + list of ranks.
- `AggregatedPoint`: A single roofline point after aggregation (has AI, FLOP/s, bandwidth).

### Backend Interface (`backends.py`)

**`ProfilerBackend`** — Session-scoped abstract base for profiler implementations (one instance per `carm profile` invocation):
- `check_prerequisites()` — One-time session probe: verify tooling availability, discover the event catalog, resolve metrics.
- `profile(run_spec, command, cwd)` — Execute the profiled command and parse its output into a `RunResult`.
- `resolved_metrics` — Metric implementations resolved for this system.
- `available_events` — Hardware events available on this system.
- `can_collect(events)` — Whether an event set fits in one run.
- `run_method_name` — Human-readable label for metadata.

**`RunSpec`** — Per-run value object: output directory + requested events.
**`RunResult`** — Per-run outcome: exit code + parsed ranks.

**`create_backend(config, resolution_cfg)`** — Factory that dispatches on `config.backend`:
- `BackendType.PAPI` → `PAPIHLBackend`
- `BackendType.PERF` → `PerfBackend`

### Backend Implementations

**`PAPIHLBackend`** (`papi_backend.py`):
- Runs user command with `PAPI_HL_OUTPUT_DIR` set.
- Checks for PAPI availability: locates `libpapi.so` via `ldconfig -p` / `pkg-config`, verifies the `PAPI_hl_region_begin` symbol via `nm`, and falls back to the `papi_hl_output_writer` utility.
- `check_prerequisites()` raises `UserError` when `/proc/sys/kernel/perf_event_paranoid` exceeds 2; an unreadable or non-integer value warns and proceeds optimistically.
- Parses `rank_{N}.json` files from `papi_hl_output/` subdirectory.

**`PerfBackend`** (`perf_backend.py`):
- Wraps command in `perf stat -x,` with auto-resolved events.
- Supports full-run and interval (`-I`) sampling modes.
- Parses `perf_stat.csv` output into a single rank.
- `can_collect()` runs a short `perf stat` probe (`sleep 0.05`) and returns False when the kernel would time-multiplex events or leave any uncounted, so `partition_events` can split the pool — perf never multiplexes.
- `check_prerequisites()` pre-flights the resolved set with the same probe and warns when it does not fit.
- `check_prerequisites()` raises `UserError` when `/proc/sys/kernel/perf_event_paranoid` exceeds 2; an unreadable or non-integer value warns and proceeds optimistically (shared `check_perf_event_paranoid()`).
- `_parse_output()` verifies the run's CSV shows no `<not counted>`/`<not supported>` rows and 100% counted time, warning post-hoc otherwise.

### Metric Resolution (`shared.py`, `papi_metrics.py`, `perf_metrics.py`)

Each backend has its own metric definitions registry (built at module load time). The generic `resolve_metrics()` in `shared.py` selects the highest-priority implementation whose required events are available. Each backend's wrapper passes its own registry:

- `papi_metrics.resolve_metrics()` → wraps `shared.resolve_metrics()` with PAPI registry.
- `perf_metrics.resolve_perf_metrics()` → wraps `shared.resolve_metrics()` with perf registry.

### File Discovery and Parsing

**PAPI HL** (`papi_loader.py`): Scans a directory for files matching `rank_{N}.json`, parses them into `RankMetrics`.

**Perf** (`perf_loader.py`): Parses `perf stat -x,` CSV output (interval or full-run mode) into `RegionMetrics` / `ThreadMetrics`.

### Aggregation (`aggregation.py`)

| Strategy | Description |
|---|---|
| `global` (default) | Single point: sum(flops), sum(bytes), max(time_s) across all ranks |
| `per-rank` | One point per MPI rank for load-balance analysis |
| `per-thread` | One point per (rank, thread) pair |
| `region_merged` | One point per unique region name, summed across all ranks/threads |
| `region_per_thread` | One point per (rank, thread, region); no cross-thread aggregation |

### Output (`output.py`)

**JSONL**: Appends to ``<output_dir>/<machine_name>/applications.jsonl`` (``format_version "3.0"``) — one appended JSON line per run, embedding run metadata, the list of aggregated roofline points (threads / ranks / regions / a single global point, per aggregation mode), and optional-metric data (``optional_bytes`` / ``optional_fractions`` per point, ``optional_metrics`` per record). Cache-residency level keys are the canonical level names (``l1``/``l2``/``l3``/``dram``) or ``"<level>plus"`` grouping keys (e.g. ``l3plus`` = L3 and DRAM); the exact bucket set is derived per role map from the miss boundaries it can measure (AMD paths emit ``{l1, l2, l3plus}``, the Intel perf path ``{l1, l2, l3, dram}``) — consumers must treat the keys as opaque. This is the file the GUI consumes (``load_applications`` requires ``format_version >= "2.0"``; it passes ``optional_fractions`` through verbatim).

### Entry Point

`profile_main(config: ProfileConfig) -> int` in `__init__.py` orchestrates the pipeline:
1. List optional metrics and exit when `--list-metrics` is passed.
2. Select the backend via factory → resolve metrics + optional metrics for current system (one-time session probe).
3. Build the required event pool. With `--merge-runs`, partition it into feasible runs (`partition_events`); without it, use the whole pool as one chunk. One app run per chunk via the same backend with a per-run `RunSpec` (own workspace subdir + event set).
4. Parse each run (inside `profile`) and merge at the raw region-counter level (`merge_runs`).
5. Coverage-check required events (`missing_required_events`) and warn on gaps.
6. Aggregate (flops/bytes + optional metrics) & write outputs (one JSONL record).

## CLI Usage

```bash
carm profile --help
```

```
usage: carm profile [-h] [--backend {papi,perf}] [--metrics {cache-residency} [{cache-residency} ...]]
  [--list-metrics] [--merge-runs] [--perf-interval PERF_INTERVAL]
  [--aggregation {global,rank,thread,region_merged,region_per_thread}]
  [--machine-name MACHINE_NAME] [--app-name APP_NAME] [--output-dir OUTPUT_DIR]
  [--keep-artifacts] [--no-papi-cache]
  [--data-type {f32,f64}] [--isa ISA [ISA ...]]
  [--verbose [{0,1,2,3,4}]]
  [command ...]

Profile instrumented applications to compute roofline metrics (AI, GFLOP/s, bandwidth).
Supports MPI, threaded, and hybrid applications.
Usage: carm profile [options] -- <command>
```

The CLI is metric-centric: FLOPS and BYTES are always collected (needed for
roofline plotting); optional metrics (e.g. ``--metrics cache-residency``) are
selected by name and resolved to event sets by the backend. By default a single
run collects the whole event set; events that could not be counted (or were
time-multiplexed) are reported in a warning. With ``--merge-runs``, when the
required events cannot all be counted together (PAPI: ``papi_event_chooser``;
perf: short probe), the app runs once per maximal feasible event partition and
the runs merge at the raw per-region counter level before metric computation —
one JSONL record results. ``--list-metrics`` prints the available optional
metrics and exits.

### Examples

**PAPI HL profiling (default):**
```bash
carm profile -- ./my_app
```

**MPI application with PAPI:**
```bash
carm profile --app-name my_simulation -- mpirun -np 8 ./my_app
```

**Perf profiling (full-run):**
```bash
carm profile --backend perf -- ./my_app
```

**Perf profiling (interval sampling):**
```bash
carm profile --backend perf --perf-interval 100 -- ./my_app
```

**Specify ISA vector widths for custom PAPI metrics:**
```bash
carm profile --data-type f64 --isa x86_avx2 x86_sse -- ./my_app
```

When `--isa` is passed, CARM builds custom FLOPS/BYTES metric definitions using only the FP_ARITH counters matching those widths (PAPI backend). Omitting `--isa` falls back to default metric resolution.

## File Schema for Instrumented Applications (PAPI HL)

Users instrument their own code with PAPI HL and emit per-rank output files. Expected format:

**Per-rank JSON** (`rank_{N}.json`):
```json
{
  "papi_version": "...",
  "cpu_info": "...",
  "max_cpu_rate_mhz": "...",
  "min_cpu_rate_mhz": "...",
  "event_definitions": {
    "PAPI_FP_OPS": {"component": "perf_event", "type": "delta"},
    ...
  },
  "threads": {
    "0": {
      "regions": {
        "0": {
          "name": "...",
          "parent_region_id": "-1",
          "cycles": "...",
          "real_time_nsec": "...",
          "PAPI_FP_OPS": "20971520",
          ...
        },
        ...
      }
    },
    ...
  }
}
```

Per-region structure includes `name`, `parent_region_id`, `cycles`, `real_time_nsec`, and raw counter values. `threads` and `regions` are dicts keyed by id string; `event_definitions` sits at the top level of the file.

For MPI rank identification in user code, recommended environment variables:
- Open MPI: `OMPI_COMM_WORLD_RANK`
- MPICH: `PMI_RANK`
- SLURM: `SLURM_PROCID`

## Design Goals

- MPI-aware by design (distinguishes ranks from threads)
- Extensible aggregation framework
- Backend-agnostic: zero per-backend branching in the orchestrator
- Each backend is a self-contained flat file
- Clear separation: config → backend → loading → aggregation → output
- GUI-compatible output via the applications JSONL format
