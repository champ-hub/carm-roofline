# Profile Module — MPI-Aware Application Profiling

The `profile` package provides a pipeline for profiling PAPI-instrumented applications, computing roofline metrics (arithmetic intensity, FLOP/s, bandwidth) with first-class support for MPI ranks and threads.

## CLI Usage

```bash
carm profile --help
```

```
usage: carm profile [-h] [--verbose [{0,1,2,3,4}]] [--aggregation {mpi-job,per-rank}]
                    [--name NAME] [--output-dir OUTPUT_DIR] [--results-dir RESULTS_DIR]
                    [command ...]

Profile instrumented applications to compute roofline metrics (AI, GFLOP/s, bandwidth).
Supports MPI, threaded, and hybrid applications.
Usage: carm profile [options] -- <command>
```

### Examples

**Single-process application:**
```bash
carm profile -- ./my_app
```

**MPI application (auto-detects mpirun):**
```bash
carm profile --name my_simulation -- mpirun -np 8 ./my_app --input data.txt
```

**Per-rank aggregation for load-balance analysis:**
```bash
carm profile --aggregation per-rank -- mpirun -np 8 ./my_app
```

**With explicit launcher override:**
```bash
carm profile --mpi-launcher srun -- srun -N2 -n16 ./my_app
```

**Custom output directory:**
```bash
carm profile --output-dir /tmp/my_results -- mpiexec -np 4 ./app
```

## Architecture

### Data Model (`model.py`)

```
Run (metadata)
├── Rank 0
│   ├── Thread 0 (flops, bytes, time_s)
│   ├── Thread 1
│   └── ...
├── Rank 1
│   ├── Thread 0
│   └── ...
└── ...
```

- `ThreadMetrics`: Per-thread PAPI counters (flops, bytes, time_s) + raw counters.
- `RankMetrics`: Collection of threads for an MPI rank; provides aggregated properties (total_flops, total_bytes, max_time).
- `RunResults`: Complete profiling result with metadata + list of ranks.
- `AggregatedPoint`: A single roofline point after aggregation (has AI, FLOP/s, bandwidth).

### Backend (`backends.py`)

- **`ProfilerBackend`** — Abstract base for profiler implementations.
- **`PAPIHLBackend`** — Runs the user command with `PAPI_HL_OUTPUT_DIR` set. Checks for PAPI availability (via `papi_hl_read` or `libpapi.so`).

### File Discovery (`loaders.py`)

Scans a directory for files matching `{prefix}_rank{N}.csv` or `{prefix}_rank{N}.json`.

**CSV format** (expected header):
```
rank, thread, flops, bytes, time_s[, custom_column...]
```

**JSON format** (expected structure):
```json
{"rank": 0, "threads": [{"thread_id": 0, "flops": ..., "bytes": ..., "time_s": ...}]}
```

### Aggregation (`aggregation.py`)

| Strategy | Description |
|---|---|
| `mpi-job` (default) | Single point: sum(flops), sum(bytes), max(time_s) across all ranks |
| `per-rank` | One point per MPI rank for load-balance analysis |

### Output (`output.py`)

- **CSV**: Writes to `<output-dir>/applications/{name}_applications.csv` in the legacy applications format consumed by the GUI.
- **JSON**: Writes to `<output-dir>/profile/{name}_profile.json` with full rank/thread hierarchy plus aggregated view.

### Entry Point

`profile_main(config: ProfileConfig) -> int` in `__init__.py` orchestrates the pipeline:
1. Run command via backend → PAPI output files
2. Discover and parse rank files
3. Build `RunResults` hierarchy
4. Aggregate & write outputs

## File Schema for Instrumented Applications

Users instrument their own code with PAPI HL and emit per-rank output files. Expected format:

**Per-rank CSV** (`{app_name}_rank{N}.csv`):
```csv
rank, thread, flops, bytes, time_s
0, 0, 1.5e10, 8.0e8, 12.34
0, 1, 1.5e10, 8.0e8, 12.35
```

**Per-rank JSON** (`{app_name}_rank{N}.json`):
```json
{
  "rank": 0,
  "threads": [
    {"thread_id": 0, "flops": 1.5e10, "bytes": 8.0e8, "time_s": 12.34},
    {"thread_id": 1, "flops": 1.5e10, "bytes": 8.0e8, "time_s": 12.35}
  ]
}
```

For MPI rank identification in user code, recommended environment variables:
- Open MPI: `OMPI_COMM_WORLD_RANK`
- MPICH: `PMI_RANK`
- SLURM: `SLURM_PROCID`

## Design Goals

- MPI-aware by design (distinguishes ranks from threads)
- Extensible aggregation framework
- Clear separation: config → backend → loading → aggregation → output
- Launcher-agnostic (mpirun, mpiexec, srun, jsrun, aprun)
- GUI-compatible output via applications CSV format
