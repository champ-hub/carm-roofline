---
title: Commands
nav_order: 4
parent: Home
has_children: true
---

# CARM subcommands

The CARM Tool provides four subcommands covering benchmarking, profiling, visualization, and result conversion, plus a Paraver trace-analysis mode inside the GUI. See each page for detailed usage instructions and examples.

---

## [Benchmarking](benchmark)

Measure peak arithmetic performance (GFLOP/s) and memory bandwidth across cache levels to construct a Cache-Aware Roofline Model. Supports multiple ISAs, data types, thread counts, and test types.

```bash
carm benchmark --test roofline
```

---

## [GUI](gui)

Launch the interactive Dash + Plotly dashboard for exploring roofline plots and application performance data.

```bash
carm gui
```

---

## [Profiling](profile)

Profile instrumented applications (MPI, threaded, hybrid) to compute roofline metrics (arithmetic intensity, FLOP/s, and memory bandwidth) using PAPI or perf backends.

```bash
carm profile [options] -- <command>
```

---

## Conversion

Convert legacy application or two-header roofline CSV files into GUI-discoverable JSONL, or export JSONL back to CSV:

```bash
carm convert legacy-results --from legacy-csv --to jsonl
carm convert results --from jsonl --to legacy-csv --output-dir restored
```

The output directory defaults to CARM's user data directory; use `--output-dir` to choose another destination.

Use `--dry-run` to validate and preview outputs, and `--overwrite` to replace existing destination files.

---

## [Paraver](paraver)

Analyze Paraver traces on the roofline — a mode of the GUI (not a separate subcommand) that plots every traced burst and exports the metrics back as Paraver-importable windows.

```bash
carm gui --paraver-trace trace.prv --paraver-window-csv window.csv
```

See the [Paraver section](paraver) for trace requirements, usage, GUI features, and performance guidance.
