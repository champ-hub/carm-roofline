---
has_children: true
title: Home
nav_order: 1
description: CARM-Roofline — Cache-Aware Roofline Model benchmarking, profiling and visualization toolkit
---

# The CARM Tool

**The CARM Tool** is a micro-benchmarking toolkit that constructs [Cache-Aware Roofline Model (CARM)](https://ieeexplore.ieee.org/document/6506838/) performance models across multiple CPU architectures (x86, ARM, RISC-V) and GPU platforms (ROCm, CUDA). It measures arithmetic performance and memory bandwidth of all cache levels to model the architecture's performance and guide optimization.

The toolkit provides:

- **Benchmarking**: generates roofline plots by measuring peak performance (FLOP/s) and peak memory bandwidth for each cache level
- **Profiling**: profiles instrumented applications to compute roofline metrics (arithmetic intensity, FLOP/s)
- **GUI**: interactive Dash + Plotly dashboard for exploring roofline plots and application data

## Quick Start

```bash
pip install carm-roofline

# Run a single-thread roofline benchmark (auto-detects ISA)
carm benchmark

# Launch the GUI to visualize results
carm gui
```

See the [Installation](installation) page for detailed setup instructions.

## Table of Contents

| Page | Description |
|------|-------------|
| [Installation](installation) | Requirements, pip install, virtual environment, from-source install |
| [Benchmark](benchmark) | Run benchmarks, argument reference |
| [GUI](gui) | Launch the interactive dashboard, argument reference |
| [Profile](profile) | Profile applications, argument reference |
