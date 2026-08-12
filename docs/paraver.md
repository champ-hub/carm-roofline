---
title: Paraver
nav_order: 2
has_children: true
---

# Paraver trace analysis

The CARM Tool can analyze **Paraver** traces. It plots every burst of a [Paraver](https://tools.bsc.es/paraver) trace as a point on the roofline: the GUI runs Paraver's `paramedir` tool over the trace, reads the hardware-counter data it extracts, and computes roofline metrics (arithmetic intensity, FLOP/s, bandwidth) per burst. The results can be exported back as Paraver-importable windows.

## Quick Start

```bash
# Install the CARM Tool with GUI extras
pip install "carm-roofline[gui]"

# Add Paraver's bin directory to your PATH (provides paramedir)
export PATH=/path/to/paraver/bin:$PATH

# Analyze a trace (or call from Paraver's context menu for full integration)
carm gui --paraver-trace trace.prv --paraver-window-csv window.csv
```

## Prerequisites

- **The CARM Tool with GUI extras** — `pip install "carm-roofline[gui]"`. See the [Installation](installation) page for the full instructions.
- **Paraver** (version 4.12 or later) and **Extrae** — BSC's trace visualization and generation tools. Extrae instruments the application and produces the trace.
- **`paramedir` on `PATH`** — the CARM Tool uses Paraver's `paramedir` tool to extract counter data from the trace. Add Paraver's `bin/` directory to your `PATH` (e.g. `export PATH=/path/to/paraver/bin:$PATH`).
- **A Paraver trace (`.prv`)**, optionally the respective **window CSV**

## CLI flags

Paraver mode is a mode of `carm gui`, enabled by giving a trace:

| Flag | Meaning |
|------|---------|
| `--paraver-trace <path>` | Path to the `.prv` trace; enables Paraver GUI mode |
| `--paraver-window-csv <path>` | Path to the window/mask CSV (**required** when `--paraver-trace` is given) |

The usual GUI options (`--gui-host`, `--gui-port`, `--results-dir`, `--verbose`) combine with Paraver mode.

## Contents

| Page | Description |
|------|-------------|
| [Trace Requirements](trace-requirements) | Which hardware counters the trace must include |
| [Usage](usage) | Step-by-step workflow: launch from Paraver or the CLI |
| [Features](features) | GUI controls, filtering, coloring, and exports |
| [Performance](performance) | Keeping the GUI responsive with large traces |
