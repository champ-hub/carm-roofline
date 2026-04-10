# CARM Roofline — Legacy Utility Scripts Reference

> **For AI agents:** This document is a technical reference for standalone analysis scripts, shared utility modules, and the results visualization stack in the CARM roofline tool. All paths are relative to the workspace root `/home/alexandre/Desktop/carm-roofline/`.

---

## Section 1: Standalone Analysis Scripts

### `calc_stream.py`

**Purpose:** Offline calculator that derives GFLOP/s performance for STREAM benchmark kernels from hardcoded pre-measured bandwidth values. Not a runnable script with CLI arguments — it is a data scratchpad.

**Inputs:** Hardcoded constants at the top of the file:
- `SIZE = 100000000` — array element count
- `apps` list of tuples: `(app_name, bytes_per_loop, ai, bandwidth_MB_per_s)`

```python
apps = [
    ("stream_ONLY_SCALE", 2 * 8, 1 / 16, 256657.042),
    ("stream_ONLY_ADD",   3 * 8, 1 / 24, 280014.001),
    ("stream_ONLY_TRIAD", 3 * 8, 1 / 12, 292077.401),
]
```

**Calculations performed per entry:**
1. `total_bytes = SIZE * bytes_per_loop`
2. `ops = ai * total_bytes`
3. `time = total_bytes / (bandwidth * 1e6)` — converts MB/s → B/s
4. `gflops = ops / time / 1e9`

**Output:** Prints one CSV row per app to stdout with columns:
```
date, method, app_name, "", precision, threads, ai, gflops, bandwidth, time
```
Example: `2026-02-19 15:47:49,PMU,stream_ONLY_SCALE,,dp,1,0.0625,3.14...,256657.0,0.00...`

**CLI:** None. Run with `python calc_stream.py`; output is always stdout.

**Intended use:** Feed its output rows into `carm_results/applications/<machine>_applications.csv` for display in `gui/dashboard.py`.

---

### `plot_memory_bandwidth.py`

**Purpose:** Plots memory bandwidth vs. buffer size using matplotlib, overlaying two data sources on a single log-scale x-axis chart.

**Input formats:**

1. **Table format** (new CARM memory sweep output — pipe `│` or `|` delimited):
   ```
   │ x86_avx2 │     6.4 KiB │ L1          │ 620.03 GB/s │
   │ x86_avx2 │     9.0 KiB │ L1          │ 619.76 GB/s │
   ```
   Columns: `isa`, `size`, `cache_level`, `bandwidth`. Size strings like `6.4 KiB`, `129.0 MiB` are parsed via `parse_human_size()`. Bandwidth is the numeric prefix of the GB/s string.

2. **Legacy format** (old memory sweep stderr lines):
   ```
   Size (per thread): 262144 Kb | Gbps: 13.259 | Instructions per Cycle: 0.08919
   ```
   Size is in Kb (converted to bytes); bandwidth is Gbps.

**Key functions:**
- `parse_table(lines, verbose=False) -> tuple[list[float], list[float]]` — returns `(sizes_bytes, bw_GB_s)`
- `parse_legacy(lines, verbose=False) -> tuple[list[float], list[float]]` — returns `(sizes_bytes, bw_Gbps)`
- `parse_human_size(s: str) -> float` — parses strings like `"6.4 KiB"` to bytes; supports `KiB`, `MiB`, `GiB`
- `main()` — argument parsing and plot generation

**CLI:**
```
plot_memory_bandwidth.py [table_file] [--legacy FILE] [--output FILE] [-v]

positional arguments:
  table_file          pipe-delimited table file; omit or use - for stdin

optional arguments:
  --legacy / -l FILE  legacy Size/Gbps file
  --output / -o FILE  output image file (png, svg, etc.)
  -v / --verbose      print per-line diagnostic messages to stderr
```

**Output:** matplotlib figure with log x-axis (bytes), linear y-axis (bandwidth). Saved to `--output` path if provided; otherwise displayed interactively (skips display silently under non-interactive Agg backend).

**Example:**
```bash
python plot_memory_bandwidth.py plot_m_new.txt --legacy plot_m_old.txt --output p.png
```

---

### `plot_timing_distribution.py`

**Purpose:** Parses benchmark debug timing output and produces per-benchmark histogram plots with summary statistics.

**Input format:** Lines matching the pattern:
```
<benchmark_name> times (ms): 120.341, 121.023, 119.987, ...
```
The benchmark name is a `\w+.*?` group; values are comma-separated floats.

**Input source:** Either a filename passed as `sys.argv[1]` or stdin (useful for piping benchmark stderr).

**Key functions:**
- `parse_timing_data(input_source) -> dict[str, list[float]]`
  - `input_source`: filename string or file-like stdin object
  - Returns `defaultdict` mapping benchmark name → list of millisecond timings
- `plot_distributions(data: dict[str, list[float]]) -> None`
  - Computes per-benchmark: mean, min, max, range, stddev
  - Plots histogram grid: `min(3, N)` columns, auto rows; subplots sized `15 × (5 * rows)` inches
  - Adds vertical lines for mean (red dashed), min (green dashed), max (orange dashed)
  - Prints statistics table to stdout
  - Always saves to `timing_distribution.png` (dpi=100) in the current working directory
  - Calls `plt.show()` after saving

**CLI:**
```
# Read from file:
python plot_timing_distribution.py debug.log

# Read from stdin (pipe benchmark output):
./benchmark --freq 4 --threads 4 2>&1 | python plot_timing_distribution.py
```

**Output file:** Always `./timing_distribution.png` (hardcoded, not configurable). Prints statistics to stdout.

---

### `array_pattern_cmp.c`

**Purpose:** C microbenchmark that compares the effective memory bandwidth of two array kernel patterns using AVX2 intrinsics, sized to be DRAM-bound by default (arrays > L3).

**What it tests:**
1. **Element-wise add** — `c[i] = a[i] + b[i]` using `_mm256_load_pd` + `_mm256_add_pd` + `_mm256_store_pd`; reads a, b; writes c. Reads/writes 3 arrays × N doubles.
2. **Interleaved update** — on a single contiguous array `x` of length `3*N`: for every 12 elements, sums `x[i..i+3]` + `x[i+4..i+7]`, stores into `x[i+8..i+11]`. Same total data volume.

**Default parameters:**
- `N = 1 << 22 = 4,194,304` elements → combined array size ≈ 96 MiB (> L3)
- `reps = 10` (best-of-N timing)

**CLI:**
```bash
# Build:
gcc -O3 -march=native -std=c11 -o array_pattern_cmp array_pattern_cmp.c

# Run (defaults):
./array_pattern_cmp

# Run with custom N and reps:
./array_pattern_cmp 6000000 5
```

**Output:** Printed to stdout:
```
N = 4194304 (elements)  -- arrays total = 96.00 MiB
reps = 10

Results (best of 10 reps):
element-wise : time = 0.057123 s, Bandwidth = 140.42 GB/s, checksum = 1.234568e+09
interleaved  : time = 0.058432 s, Bandwidth = 137.26 GB/s, checksum = 9.876543e+08
```

**Bandwidth formula:** `(N * 3 * sizeof(double)) / best_time / 1e9` GB/s (same for both kernels).

**Cache-flush mechanism:** `flush_caches()` touches arrays at 64-element stride before each rep to evict from L1/L2/L3.

**Requirements:** x86 with AVX2 support; `gcc` with `-march=native`; `immintrin.h`.

---

## Section 2: Shared Utility Modules

### `utils.py`

**Purpose:** Shared utility library used by legacy scripts (`run.py`, analysis calculators) and `gui/dashboard.py`. Provides config file I/O, math helpers, legacy roofline data parsing, and matplotlib-based roofline plotting.

**Configuration file:** `./config/auto_config/config.txt` (key=value format)

**Key functions:**

| Function | Signature | Description |
|---|---|---|
| `read_library_path` | `(tag: str) -> str \| None` | Read a `tag=value` entry from config file |
| `write_library_path` | `(tag: str, path: str) -> None` | Append a `tag=value` line to config file |
| `make_power_of_two_ticks` | `(min_val, max_val) -> tuple[list, list]` | Generate log2 tick values + HTML superscript labels for Plotly axes |
| `ensure_list` | `(marker_dict, attr_name, default_value, n_points) -> list` | Coerce a marker attribute to a list of length n_points |
| `custom_round` | `(value, digits=4) -> float` | Smart rounding: uses extra precision for sub-1 values to preserve significant digits |
| `positive_int` | `(value) -> int` | argparse validator: raises `ArgumentTypeError` if not a positive integer |
| `round_power_of_2` | `(number) -> int` | Returns the next power of 2 above `number` |
| `carm_eq` | `(ai, bw, fp) -> np.ndarray` | Roofline equation: `min(ai * bw, fp)` using numpy |
| `parse_title_line` | `(line: str) -> dict` | Parses legacy `.out` file header line into dict with keys: `name`, `isa`, `precision`, `threads`, `load`, `store`, `inst` |
| `read_roofline_data` | `(filename: str) -> tuple[dict, dict, dict]` | Reads legacy `.out` roofline result file → `(title, data, data_cycles)` where data keys are `L1`, `L2`, `L3`, `DRAM`, `FP`, `FP_FMA` |
| `read_data_from_files` | `(directory: str, autochoice: int) -> tuple[dict, dict, dict]` | Lists `carm_results/roofline/*.out` files, prompts user (or auto-selects), returns parsed data |
| `plot_roofline_with_dot` | `(executable_path, exec_flops, exec_ai, choice, roi, date, method)` | Generates full matplotlib roofline SVG saved to `carm_results/applications/<name>_<...>.svg` |
| `update_csv` | `(machine, executable_path, exec_flops, exec_ai, bandwidth, time, name, date, isa, precision, threads, method, VLEN, LMUL)` | Appends or creates `carm_results/applications/<machine>_applications.csv` |

**Who uses `utils.py`:**
- `gui/dashboard.py` (imports as `ut`): `carm_eq()`, `make_power_of_two_ticks()`, `ensure_list()`
- `gui/gui_utils.py` (imports as `ut`): `carm_eq()`
- Legacy analysis calculators (`PMU_AI_Calculator.py`, `DBI_AI_Calculator.py`): `plot_roofline_with_dot()`, `update_csv()`

**Legacy `.out` file format** (read by `read_roofline_data`):
```
<name> ISA <isa> <precision> <threads> Load <n> Store <n> <inst>
L1: 620.5
L2: 420.3
L3: 180.1
DRAM: 45.2
FP: 512.0
FP_FMA: 1024.0
L1 Instruction Per Cycle: 2.5
...
```

**CSV application file format** (written by `update_csv`):
```
Date,Method,Name,ISA,Precision,Threads,AI,Gflops,Bandwidth,Time
2026-02-19,PMU,stream_ONLY_SCALE,avx2,dp,1,0.0625,3.14,256657.0,0.0039
```

## Section 3: Results Visualization

### `gui/dashboard.py`

**Purpose:** Interactive web dashboard (~3614 lines) for CARM roofline visualization, cross-machine comparison, and on-demand benchmark execution. The primary user-facing interface for the tool.

**Stack:**
- **Dash** (Plotly) — reactive web framework
- **dash-bootstrap-components** — Bootstrap 5 styling
- **dash-daq** — toggle switches
- **diskcache** — background callback state (`./cache/`)
- **Plotly `go.Figure`** — roofline chart rendering
- **Pandas** — data filtering/querying

**Launch:**
```bash
python -m gui.dashboard
# Serves at http://127.0.0.1:8050 (default Dash port)
# debug=False (hardcoded)
```

**Entry point:**
```python
if __name__ == "__main__":
    app.run(debug=False)
```

**Data it reads:**
- Roofline CSV results: `./carm_results/roofline/*.csv`
  - Format (read by `gui_utils.read_csv_file`): two header rows (machine name + cache sizes in row[1], data columns in row[2]), data rows from row[3]
  - Filename convention: `<machine_name>_*.csv` — machine name extracted as `filename.split("_")[0]`
- Application results CSV: `./carm_results/applications/<machine>_applications.csv`
  - Format (read by `gui_utils.read_application_csv_file`): `Date,Method,Name,ISA,Precision,Threads,AI,GFLOPS,Bandwidth,Time`

**ISA options:** Auto-detected from `platform.machine()`:
- `x86_64` → AVX512, AVX2, SSE, Scalar
- `aarch64` → SVE, NEON, Scalar
- `riscv64` → RVV1.0, RVV0.7, Scalar

**UI layout (`app.layout = dbc.Container(...)`):**
- Top navbar with "CARM Tool Functions" offcanvas (sidebar) and "Graph Customization" offcanvas (sidebar2)
- Central `dcc.Graph(id="graphs")` — the roofline plot
- Machine selector dropdown (`id="filename"`) populated from `./carm_results/roofline/`
- Filter controls: ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FP instruction, Date

**Sidebars:**
1. **CARM Functions sidebar** (`id="offcanvas"`) — benchmark configuration and launch:
   - Machine name, cache sizes (L1/L2/L3 Kb), thread counts, NUMA interleaving
   - ISA extensions checkboxes (arch-specific), precision (DP/SP), load/store ratio
   - DRAM test size; "Run CARM Benchmarks" and "Run Application Analysis" buttons
   - "Stop Benchmark/Analysis" cancel button

2. **Graph Customization sidebar** (`id="offcanvas2"`) — display settings:
   - Exponent notation toggle, graph dimensions (width/height), line/dot sizes
   - Font sizes (title, axis, legend, tick, annotation, tooltip)
   - Annotation visibility and angle controls
   - Legend toggle and detail level (compact vs. full bandwidth values)

**Default graph dimensions:**
```python
DEFAULTS = {
    "graph-width": 1900, "graph-height": 690,
    "line-size": 3, "dot-size": 10, "title-size": 20,
    "axis-size": 20, "legend-size": 13, "tick-size": 18,
    "annotation-size": 12, "tooltip-size": 14,
}
```

**Key Dash callbacks (50+ total):**
- Machine selection → populate filter dropdowns, load CSV, render roofline
- Filter changes → recompute `calculate_roofline()` + `plot_roofline()` → update `dcc.Graph`
- "Run CARM Benchmarks" → background callback calling `run.py` benchmark pipeline
- "Run Application Analysis" → background callback calling `PMU_AI_Calculator` or `DBI_AI_Calculator`
- Annotation accordion → per-level visibility and angle inputs
- Point style editor → click on point → color/size/symbol picker

**Imports from project:**
```python
import DBI_AI_Calculator   # application DBI analysis
from . import gui_utils as gut  # CSV reading, roofline calculation, Plotly traces
import PMU_AI_Calculator   # application PMU analysis
import run                 # legacy benchmark execution
import utils as ut         # carm_eq(), make_power_of_two_ticks(), ensure_list()
```

---

### `gui/gui_utils.py`

**Purpose:** Helper module for `gui/dashboard.py` containing all CSV I/O, roofline geometry computation, and Plotly trace/annotation generation. Not standalone.

**Key functions:**

#### CSV I/O

`read_csv_file(file_path: str) -> tuple[str, int, int, int, list[dict]]`
- Reads roofline CSV from `./carm_results/roofline/`
- **CSV format:** Row 0 = `["", machine_name, "", l1_size, "", l2_size, "", l3_size, ...]`; Row 1 = column headers; Rows 2+ = data
- Returns: `(machine_name, l1_size_kb, l2_size_kb, l3_size_kb, data_list)`
- Each data dict keys: `Date`, `ISA`, `Precision`, `Threads`, `Loads`, `Stores`, `Interleaved`, `DRAMBytes`, `FPInst`, `L1`, `L2`, `L3`, `DRAM`, `FP`, `FP_FMA` (floats from specific column indices in the CSV)

`read_application_csv_file(file_path: str) -> list[dict] | bool`
- Reads `carm_results/applications/<machine>_applications.csv`
- Returns list of dicts with keys: `Date`, `Method`, `Name`, `ISA`, `Precision`, `Threads`, `AI` (float), `GFLOPS` (float), `Bandwidth` (float), `Time` (float)
- Returns `False` on missing file, empty file, or parse error

#### Roofline Geometry

`calculate_roofline(values: list, min_ai: float) -> dict[str, dict]`
- `values` order: `[L1_bw, L2_bw, L3_bw, DRAM_bw, FP_peak, FP_FMA_peak, inst_label_str]`
- Computes ridge points and bandwidth segment endpoints for each cache level over `ai = linspace(min(0.00390625, min_ai), 256, 200000)`
- Returns `dots` dict: keys are cache level names + `values[6]` (FP inst label)
  ```python
  dots["L1"] = {
      "start": [ai_start, gflops_start],
      "mid":   [ai_mid,   gflops_mid],    # geometric mean of start/ridge
      "ridge": [ai_ridge, gflops_ridge],  # transition point
      "end":   [ai_end,   gflops_end],
  }
  ```

#### Plotly Trace Generation

`plot_roofline(values, dots, name_suffix, ISA, line_legend, line_size, line_legend_detailed) -> list[go.Scatter]`
- `name_suffix=""` → black lines; `name_suffix` non-empty → red lines
- Generates one `go.Scatter` trace per cache level (L1=solid, L2=solid, L3=dash, DRAM=dot)
- Adds FP and FP_FMA peak traces (dashdot and solid respectively)
- `line_legend_detailed=True` → labels include bandwidth values (e.g., `"L1 AVX2 Bandwidth: 620 GB/s"`)
- Returns list of Plotly trace objects ready to pass to `go.Figure.add_traces()`

`draw_annotation(values, lines, name_suffix, ISA, cache_level, graph_width, graph_height, anon_size, x_range=None, y_range=None) -> go.layout.Annotation`
- Positions annotation label on the bandwidth slope line using pixel-space angle calculation
- `cache_level` ∈ `{"L1", "L2", "L3", "DRAM", "FMA", "FP"}`
- Angle computed from log10 pixel coordinates: `math.degrees(math.atan(pixel_slope))`
- Returns a single `go.layout.Annotation` object

#### Helpers

`construct_query(ISA, Precision, Threads, Loads, Stores, Interleaved, DRAMBytes, FPInst, Date) -> str | None`
- Builds a pandas `.query()` string from non-None filter arguments
- Returns `None` if all arguments are falsy (no filter)

`interpolate_color(start_color, end_color, factor) -> str`
- Linear RGB interpolation; returns `"rgb(r, g, b)"`; inputs are `(r, g, b)` integer tuples

`extract_last_segment(s: str) -> str`
- Returns `s.split("_")[-1]` if underscore present, else `s`

`extract_prefix(s: str) -> str`
- Returns `s.rsplit("_", 1)[0]` if underscore present, else `s`

**Dependencies:** `csv`, `os`, `math`, `plotly.graph_objects`, `numpy`, `utils as ut`
