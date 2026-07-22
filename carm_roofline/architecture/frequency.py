"""Utilities for setting CPU frequency via sysfs.

These helpers are Python equivalents to the legacy C frequency setter and
are useful when you do not want to rebuild or run the autoconfig helper.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


def _iter_scaling_max_freq_files() -> Iterable[Path]:
    for cpu_dir in SYS_CPU.glob("cpu[0-9]*"):
        candidate = cpu_dir / "cpufreq" / "scaling_max_freq"
        if candidate.exists():
            yield candidate


def set_cpu_frequency(new_max_freq_khz: int) -> None:
    """Set max frequency for all CPUs (requires root on most systems)."""
    if new_max_freq_khz <= 0:
        raise ValueError("new_max_freq_khz must be positive")

    errors = []
    for path in _iter_scaling_max_freq_files():
        try:
            with path.open("w") as f:
                f.write(str(new_max_freq_khz))
        except OSError as exc:
            errors.append((path, exc))

    if errors:
        msgs = ", ".join(f"{p}: {e}" for p, e in errors)
        raise OSError(f"Failed to set frequency on: {msgs}")


def read_cpu_frequencies() -> dict[str, str]:
    """Read current max frequencies (for verification/debug)."""
    freqs: dict[str, str] = {}
    for path in _iter_scaling_max_freq_files():
        try:
            freqs[path.parent.parent.name] = path.read_text().strip()
        except OSError:
            freqs[path.parent.parent.name] = "<unreadable>"
    return freqs
