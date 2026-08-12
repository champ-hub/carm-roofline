---
title: Usage
parent: Paraver
nav_order: 2
---

# Usage

There are two ways to open a trace in the CARM GUI: from Paraver itself, or from the command line.

## Launching from Paraver

1. **Generate a trace with Extrae** — instrument your application and produce a Paraver trace. Your Extrae configuration must include the required hardware counters — see [Trace Requirements](trace-requirements).
2. **Load the trace in Paraver** and zoom into a section of interest.
3. **Right-click the timeline and select the option to launch the CARM GUI**, then click **Run** in the dialog.
4. **Open the GUI** in your browser at the URL printed in the Paraver console.

## Launching from the command line

Export the window you want to analyze from Paraver as a CSV, then run:

```bash
carm gui --paraver-trace trace.prv --paraver-window-csv window.csv
```

## What You'll See

The GUI displays:

- The **architecture's roofline** (peak performance bounds for compute and memory)
- **Your application's bursts** as points on the roofline plot
- Each point's position is determined by its **performance** (FLOP/s) and **arithmetic intensity** (FLOP/byte)

The position of your points on the roofline helps identify bottlenecks and optimization opportunities. See [Features](features) for details on filtering, coloring, and exports.

## Troubleshooting

- **`paramedir` not found** — add Paraver's `bin/` directory to your `PATH`. See the [Paraver](paraver) introduction.
- **Missing legend file** — a color-based window needs its `.legend.csv` file next to the window CSV.
- **Missing counters** — verify that the trace includes the required hardware counters. See [Trace Requirements](trace-requirements).
- **Load failures** — the GUI shows a warning and continues with the roofline only.
