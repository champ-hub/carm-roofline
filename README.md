
# The CARM Tool


Cache-Aware Roofline Model (CARM) toolkit - benchmark, visualize and profile Intel, AMD, ARM, RISC-V CPUs and NVIDIA, AMD GPUs.

The package provides:

- automatic benchmarking for supported ISAs
- arithmetic, memory, and roofline benchmark modes
- optional web GUI for interactive result visualization

> [!WARNING]
> The development branch and associated python package are still experimental, and do not include all features yet. For a stable release, please refer to the main branch.

## Installation

Install from PyPI:

```bash
# command-line interface only:
pip install carm-roofline
# with GUI extras:
pip install carm-roofline[gui]
# with all extras (GUI + development tools):
pip install carm-roofline[all]
```

To avoid conflicts with other packages, we recommend installing in a virtual environment:

```bash
python -m venv carm-env
source carm-env/bin/activate
pip install carm-roofline[all]
```

## Requirements

- Python 3.9+
- GCC (for compiling benchmark binaries)

## Command-Line Interface

The installed command is:

```bash
carm
```

Available subcommands:

- `benchmark` - run benchmark suites
- `gui` - launch the results dashboard
- `profile` - coming soon

Show help:

```bash
carm --help
carm benchmark --help
carm gui --help
```

## Basic Usage

Run a default benchmark (roofline mode with auto-detected settings):

```bash
carm benchmark
```

Other usage examples:

```bash
# Short roofline benchmark (1 second per micro-bench)
carm benchmark --test roofline --test-time 1
# Arithmetic-only
carm benchmark --test arithmetic --instruction add --data-type f32
# Memory-only
carm benchmark --test memory --mem-target L2 --ld-st-ratio 2:1
# Dry run (generate benchmark code, skip compile/execute):
carm benchmark --dry-run --test arithmetic --test-time 1 --verbose 4
```

## Output and Results

By default, results are written under `carm_results`.

Control output directory and format(s):

```bash
carm benchmark --output-dir carm_results --output-fmt table json csv plot
```

Use `--name` to label result files:

```bash
carm benchmark --name my_machine
```

## GUI Mode

Launch the dashboard (requires GUI extras):

```bash
carm gui
```

Custom results location and port:

```bash
carm gui --results-dir carm_results --gui-port 8050
```

## Publications and Citation

If you use the CARM and the CARM Tool in papers or reports, please cite:

<p>
  <a href="https://doi.org/10.1109/L-CA.2013.6" alt="Publication">
    <img src="https://img.shields.io/badge/DOI-10.1109/L--CA.2013.6-blue.svg"/></a>
</p>

<p>
  <a href="https://doi.org/10.1016/j.future.2020.01.044" alt="Publication">
    <img src="https://img.shields.io/badge/DOI-10.1016/j.future.2020.01.044-blue.svg"/></a>
</p>

J. Morgado, L. Sousa, A. Ilic. "CARM Tool: Cache-Aware Roofline Model Automatic Benchmarking and Application Analysis", IEEE International Symposium on Workload Characterization (IISWC), Vancouver, British Columbia, Canada, 2024.

A. Ilic, F. Pratas and L. Sousa, "Cache-aware Roofline model: Upgrading the loft," IEEE Computer Architecture Letters, vol. 13, no. 1, pp. 21-24, Jan.-June 2014. doi:10.1109/L-CA.2013.6.

D. Marques, A. Ilic, Z. A. Matveev, and L. Sousa, "Application-driven cache-aware roofline model," Future Generation Computer Systems, vol. 107, pp. 257-273, 2020. doi:10.1016/j.future.2020.01.044.
