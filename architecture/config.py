"""TOML-based architecture configuration system for memory hierarchy.

This module provides a unified way to specify cache hierarchy configuration
instead of scattered CLI arguments. The TOML format is human-friendly,
explicitly documents the cache topology, and enables validation.

Design principles:
  - Explicit over implicit: instances count is required, not derived
  - String sizes are human-friendly (32K, 8M, 1G) and parsed at load time
    - Top-level optional CPU metadata: total_cpus/smt_degree/cpu_offset
    - Direct mapping: TOML cache_levels -> SimpleMemoryTopology fields
    - Convention: last configured level is treated as DRAM in hierarchy iteration/naming
    - Minimal but expressive: supports sparse/non-contiguous cache levels
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from core import Bytes, UserError

from .memory import SimpleMemoryTopology

# Handle tomli import for Python < 3.11
try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ImportError:
    import tomli as tomllib


def _read_optional_int(config: dict[str, object], key: str, default: int | None = None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer, got: {value!r}")
    return value


def load_memory_topology_from_toml(config_path: Path | str) -> SimpleMemoryTopology:
    """Load memory topology specification from a TOML configuration file.

    TOML Format:
    ```toml
    total_cpus = 8          # Optional: logical CPUs available for placement
    smt_degree = 1          # Optional: SMT level (1 = no SMT, 2 = 2-way SMT, etc.)
    cpu_offset = 0          # Optional: offset for cpusets not starting at CPU 0

    [[cache_levels]]
    level = 1               # Required: numeric cache level (1, 2, 3, ...)
    instances = 8           # Required: number of cache instances at this level
    size = "32KiB"          # Required: cache size using binary prefixes (KiB, MiB, GiB, etc.)

    [[cache_levels]]
    level = 2
    instances = 4
    size = "512KiB"

    [[cache_levels]]
    level = 3
    instances = 2
    size = "8MiB"

    [[cache_levels]]
    level = 4
    instances = 1
    size = "64GiB"         # Last level is represented as DRAM
    ```

    Args:
        config_path: Path to .toml configuration file

    Returns:
        SimpleMemoryTopology instance initialized with parsed hierarchy

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If TOML is malformed, required fields missing, or validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise UserError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        raise UserError(f"Failed to parse TOML file '{config_path}': {e}") from e

    total_cpus = _read_optional_int(config, "total_cpus")
    smt_degree = cast(int, _read_optional_int(config, "smt_degree", 1))
    cpu_offset = cast(int, _read_optional_int(config, "cpu_offset", 0))

    # Extract cache levels
    cache_levels_data = config.get("cache_levels", [])
    if not isinstance(cache_levels_data, list) or not cache_levels_data:
        raise UserError("Configuration must contain at least one [[cache_levels]] section")

    # Parse and validate each cache level (processed in document order; order determines level numbers)
    level_sizes: list[Bytes] = []
    instances_per_level: list[int] = []

    for level_dict in cache_levels_data:
        if not isinstance(level_dict, dict):
            raise UserError(f"Each [[cache_levels]] entry must be a table, got: {type(level_dict).__name__}")
        try:
            instances = level_dict["instances"]
            size_str = level_dict["size"]
        except KeyError as e:
            raise UserError(f"Cache level section missing required field: {e}") from e

        if not isinstance(instances, int):
            raise UserError(f"'instances' must be an integer, got: {instances!r}")
        if not isinstance(size_str, str):
            raise UserError(f"'size' must be a string, got: {size_str!r}")

        size = Bytes.from_string(size_str)
        level_sizes.append(size)
        instances_per_level.append(instances)

    # Create and return SimpleMemoryTopology directly from TOML model.
    # The class constructor performs all semantic validation.
    return SimpleMemoryTopology(
        level_sizes=level_sizes,
        instances_per_level=instances_per_level,
        total_cpus=total_cpus,
        smt_degree=smt_degree,
        cpu_offset=cpu_offset,
    )


def emit_template_toml(output_path: Path | str) -> None:
    """Emit a template TOML configuration file for easy editing.

    Creates a well-commented template that users can customize for their system.

    Args:
        output_path: Path where the template should be written

    Raises:
        IOError: If unable to write to output file
    """
    output_path = Path(output_path)

    num_levels = 4
    num_cpus = 8

    default_sizes = ["32KiB", "256KiB", "8MiB", "64GiB"]
    default_instances = [num_cpus, max(1, num_cpus // 2), max(1, num_cpus // 4), 1]

    level_config: list[str] = []
    for level in range(1, num_levels + 1):
        index = min(level - 1, len(default_sizes) - 1)
        level_config.extend(
            [
                "[[cache_levels]]",
                f"instances = {default_instances[index]}",
                f'size = "{default_sizes[index]}"',
                "",
            ]
        )

    # Last configured hierarchy level is interpreted as DRAM.
    if level_config and level_config[-1] == "":
        level_config.pop()

    template = f"""# CARM Architecture Configuration - Memory Hierarchy Specification
#
# This file defines the cache hierarchy of your system using TOML format.
# Edit the values below to match your actual hardware configuration.
#
# If you have a standard x86 architecture and are running on Linux, you
# can safely rely on the auto-detection feature and skip manual configuration,
# and can ignore this file.
#
# Documentation:
#   - instances: Number of distinct cache instances at this level
#                (e.g., 8 cores = 8 L1 instances if L1 is per-core)
#   - size: Size of each instance using binary prefixes (KiB, MiB, GiB, TiB)
#           Examples: "32KiB", "256KiB", "8MiB", "1GiB"
# IMPORTANT: The last [[cache_levels]] entry is interpreted as DRAM.

# Total number of CPUs (threads) in the system.
# Optional, but recommended for realistic synthetic affinity planning.
total_cpus = {num_cpus}

# Simultaneous multithreading degree (1 = no SMT, 2 = 2-way SMT, etc.)
# Optional; used to prefer one thread per physical core before SMT siblings.
smt_degree = 2

# CPU offset for edge-case systems where CPU numbering doesn't start at 0.
# Optional, default is 0.
cpu_offset = 0

# ============================================================================
# Cache Hierarchy Definition
# Each [[cache_levels]] block defines one cache level.
# The final configured level is interpreted as DRAM.
# ============================================================================

{chr(10).join(level_config)}

# ============================================================================
# Examples for different system configurations:
#
# Single-socket system with 8 cores, 3-level cache + DRAM:
#   L1: 8 cores x 32 KiB (8 instances, per-core)
#   L2: 8 cores x 256 KiB (8 instances, per-core)
#   L3: shared (2 instances, each shared by 4 cores)
#   DRAM: 1 instance x 64 GiB
#
# Multi-socket system (e.g., 2 sockets x 4 cores):
#   L1: 8 cores x 32 KiB (8 instances)
#   L2: 8 cores x 256 KiB (8 instances)
#   L3: 2 instances x 8 MiB (each shared by 4 cores per socket)
#   DRAM: 1 instance (system memory)
#
# System with only L1 and DRAM (no L2/L3):
#   [[cache_levels]]
#   instances = 4
#   size = "64KiB"
#
#   [[cache_levels]]
#   instances = 1
#   size = "32GiB"
#
# ============================================================================
"""

    try:
        with open(output_path, "w") as f:
            f.write(template)
    except OSError as e:
        raise OSError(f"Failed to write template to '{output_path}': {e}") from e
