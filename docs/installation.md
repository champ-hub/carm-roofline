---
title: Installation
nav_order: 2
parent: Home
---

# Installation

## Requirements

- **Python** 3.9 or later
- **pip** (or setuptools >=64.0 + wheel for source installs)
- **C compiler**: GCC >=4.9
- **Linux** (primary target; cross-compilation available for ARM/RISC-V)

Optional dependencies:

- **GUI**: Dash, Plotly, pandas (`pip install "carm-roofline[gui]"`)
- **Profiling with PAPI**: PAPI development libraries (`libpapi` and headers) on your system

## Installing via pip

The recommended way to install CARM-Roofline is from PyPI:

```bash
pip install carm-roofline
```

### With GUI support

```bash
pip install "carm-roofline[gui]"
```

### Full installation (GUI + development tools)

```bash
pip install "carm-roofline[all]"
```

### Using a Virtual Environment

If you encounter dependency conflicts, use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install carm-roofline
```

## Installing from Source

Clone the repository and install:

```bash
git clone --recurse-submodules https://github.com/champ-hub/carm-roofline.git
cd carm-roofline
pip install -e . # -e for editable install, optional
```

For development (including dev dependencies):

```bash
pip install -e ".[all]"
```

## Verifying the Installation

Check that the `carm` command is available:

```bash
carm --help
```

You should see the top-level help screen listing the available subcommands: `benchmark`, `gui`, and `profile`.

## Next Steps

- Run your first benchmark: `carm benchmark`
- Launch the GUI: `carm gui`
- Profile an application: `carm profile -- your_app --args`
