---
title: Features
parent: Paraver
nav_order: 3
---

# Paraver GUI Features

In Paraver mode the GUI gains a **Paraver** tab in the navbar, next to **CARM View** and **Settings**. The tab groups export, filtering, and color controls; a **Time window** range slider sits above the plot.

## Time window

A range slider above the plot, in seconds, covering the trace extent. Drag the handles to zoom the displayed burst range.

## Filtering

- **Minimum arithmetic intensity** — slider; default threshold `1e-5` OPS/Byte; the leftmost position switches filtering off.
- **Minimum duration** — slider over seconds; default `100 µs`; the leftmost position switches filtering off.

Filters affect only the displayed view; exports always cover the whole trace.

## Coloring

Five modes:

- **Paraver colors** (default): the same colors as the Paraver timeline.
- **Age**, **Thread ID**, **Load/store ratio**, **ISA**: alternative per-point colorings.

## Tooltips

Hovering a point shows its arithmetic intensity, performance, bandwidth, duration, FLOPs, bytes, load/store percentage, per-ISA operation shares, the Paraver value, and the state label.

## Exports

Six export actions write CSV window files than can be opened in Paraver. To open them, click the path printed in the Paraver console.

| Action | Files | Content |
|--------|-------|---------|
| Performance (GFLOPS) | `carm_gflops.csv` | Per-burst GFLOPS |
| Arithmetic intensity | `carm_ai.csv` | Per-burst arithmetic intensity (OPS/Byte) |
| Load percentage | `carm_ldst_percent.csv` | Loads / (loads + stores) in % |
| Roof labels | `carm_roofs.csv` + `carm_roofs.legend.csv` | L1 / L2 / L3 / DRAM / No Floating Point Operations Found / Above L1 |
| Roofline region | `carm_roofline_region.csv` + `carm_roofline_region.legend.csv` | Memory Bound / Mixed / Compute Bound |
| Roof proximity | `carm_rel_l1.csv`, `carm_rel_l2.csv`, `carm_rel_l3.csv`, `carm_rel_dram.csv` | Distance-to-roof ratio 0–1, one file per roof level |

Roof-dependent exports (roof labels, region, proximity) use the roofline of the first roof card. Load the files back into Paraver (open them as new windows next to the same `.prv`) to overlay the CARM metrics on your trace.
