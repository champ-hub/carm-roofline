# Profile Module — MPI-Aware Application Profiling

The `profile` package provides a pipeline for profiling instrumented applications (PAPI HL or Linux perf), computing roofline metrics (arithmetic intensity, FLOP/s, bandwidth) with first-class support for MPI ranks and threads.

## Architecture

### Module Overview

| Module | Contents |
|---|---|
| `backends.py` | `ProfilerBackend` ABC + `create_backend()` factory |
| `papi_backend.py` | `PAPIHLBackend` — PAPI High-Level API backend |
| `perf_backend.py` | `PerfBackend` — Linux perf stat backend |
| `shared.py` | Backend-agnostic types: `MetricType`, `MetricDefinition`, `MetricResolutionConfig`, `MetricContext`, `compute_region_point()`, `sum_roofline_points()`, `resolve_metrics()` |
| `papi_metrics.py` | PAPI-specific metric definitions, event discovery, `resolve_metrics()` wrapper |
| `perf_metrics.py` | Perf-specific metric definitions, event discovery, `resolve_perf_metrics()` |
| `papi_loader.py` | PAPI HL rank-file discovery and parsing |
| `perf_loader.py` | Perf CSV output parsing |
| `aggregation.py` | Aggregation strategies (`global`, `per-rank`, `per-thread`, `region_merged`, `region_per_thread`) |
| `config.py` | `ProfileConfig` + argument parsing |
| `model.py` | Data model: `RegionMetrics`, `ThreadMetrics`, `RankMetrics`, `RunMetadata`, `RunResults` |
| `output.py` | Output writers (CSV, JSON) |

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

**`ProfilerBackend`** — Abstract base for profiler implementations:
- `check_prerequisites()` — Verify tooling availability.
- `run(command, cwd)` — Execute profiled command, produce output files.
- `resolved_metrics` — Metric implementations resolved for this system.
- `parse_output()` — Parse output files into `list[RankMetrics]`.
- `run_method_name` — Human-readable label for metadata.

**`create_backend(config, workspace, resolution_cfg)`** — Factory that dispatches on `config.backend`:
- `BackendType.PAPI` → `PAPIHLBackend`
- `BackendType.PERF` → `PerfBackend`

### Backend Implementations

**`PAPIHLBackend`** (`papi_backend.py`):
- Runs user command with `PAPI_HL_OUTPUT_DIR` set.
- Checks for PAPI availability (via `papi_hl_read` or `libpapi.so`).
- Parses `rank_{N}.json` files from `papi_hl_output/` subdirectory.

**`PerfBackend`** (`perf_backend.py`):
- Wraps command in `perf stat -x,` with auto-resolved events.
- Supports full-run and interval (`-I`) sampling modes.
- Parses `perf_stat.csv` output into a single rank.

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

**CSV**: Writes to ``<output-dir>/applications/{name}_applications.csv`` in the legacy applications format consumed by the GUI.
**JSONL**: Writes to ``<output_dir>/<name>/applications.jsonl`` — one appended JSON line per run, embedding run metadata and the list of aggregated roofline points (threads / ranks / regions / a single global point, per aggregation mode).

### Entry Point

`profile_main(config: ProfileConfig) -> int` in `__init__.py` orchestrates the pipeline:
1. Create backend via factory → resolve metrics for current system.
2. Create temporary workspace.
3. Run profiled command via backend → output files.
4. Parse output files via `backend.parse_output()` (no per-backend branching).
5. Build `RunResults` with metadata from `backend.run_method_name`.
6. Aggregate & write outputs.

## CLI Usage

```bash
carm profile --help
```

```
usage: carm profile [-h] [--backend {papi,perf}] [--papi-events PAPI_EVENTS]
  [--perf-events PERF_EVENTS] [--perf-interval PERF_INTERVAL]
  [--aggregation {global,rank,thread,region_merged,region_per_thread}]
  [--name NAME] [--output-dir OUTPUT_DIR] [--keep-artifacts]
  [--data-type {f32,f64}] [--isa ISA [ISA ...]]
  [--verbose [{0,1,2,3,4}]]
  [command ...]

Profile instrumented applications to compute roofline metrics (AI, GFLOP/s, bandwidth).
Supports MPI, threaded, and hybrid applications.
Usage: carm profile [options] -- <command>
```

### Examples

**PAPI HL profiling (default):**
```bash
carm profile -- ./my_app
```

**MPI application with PAPI:**
```bash
carm profile --name my_simulation -- mpirun -np 8 ./my_app
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
  "rank": 0,
  "threads": [
    {"thread_id": 0, "event_definitions": {...}, "regions": [...]},
    ...
  ]
}
```

Per-region structure includes `name`, `parent_region_id`, `cycles`, `time_nsec`, and raw counter values.

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
- GUI-compatible output via applications CSV format
