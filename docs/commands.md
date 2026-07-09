---
title: Commands
nav_order: 4
parent: Home
has_children: true
---

# CARM subcommands

The CARM tool provides three main subcommands covering benchmarking, profiling, and visualization. See each subcommand's page for detailed usage instructions and examples.

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
