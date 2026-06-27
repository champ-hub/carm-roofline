"""Machine identity: deterministic hardware signature and run-name generation.

Provides a shared home for both the ``benchmark`` and ``profile`` subcommands to
derive a human-readable, deterministic run name from non-measured hardware
properties (CPU model, architecture, vendor, memory topology, vector length).

Measured values such as frequencies are deliberately excluded so that the same
machine yields the same name across runs.
"""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .memory import MemoryLevelInfo, MemoryTopology, MemoryTopologyLike

if TYPE_CHECKING:
    from .architecture import Architecture


@dataclass(frozen=True)
class CpuInfo:
    """CPU identification read from /proc/cpuinfo."""

    model_name: str | None
    vendor: str | None


@dataclass(frozen=True)
class MemoryLevelSignature:
    """Deterministic signature of one memory hierarchy level."""

    name: str
    size_bytes: int
    instances: int
    num_sharing_threads: int


@dataclass(frozen=True)
class MachineSignature:
    """Deterministic hardware signature for run naming.

    Contains only non-measured, deterministic hardware properties. Excludes
    frequencies and other noisy measurements, so two runs on the same machine
    produce the same signature.
    """

    model_name: str
    arch: str
    vendor: str
    memory_levels: tuple[MemoryLevelSignature, ...]
    vector_length: int | None

    @property
    def config_hash(self) -> str:
        """Deterministic 8-char hex SHA256 of non-model-name fields.

        Hash inputs: arch, vendor, memory_levels (name, size, instances,
        sharing), vector_length. The model name is excluded because it forms the
        human-readable prefix of the run name instead.
        """
        levels_str = "|".join(
            f"{lvl.name}:{lvl.size_bytes}:{lvl.instances}:{lvl.num_sharing_threads}" for lvl in self.memory_levels
        )
        canonical = f"arch={self.arch};vendor={self.vendor};levels={levels_str};vlen={self.vector_length}"
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def read_cpuinfo() -> CpuInfo:
    """Read (model_name, vendor) from /proc/cpuinfo.

    Parses the first processor block. Returns ``CpuInfo(None, None)`` if
    /proc/cpuinfo is unavailable or cannot be parsed.

    Recognized keys:
        - x86: "model name" (model), "vendor_id" (vendor)
        - ARM 32-bit: "Hardware" (fallback model name)
        - ARM 64-bit: "CPU implementer" (vendor)
    """
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return CpuInfo(model_name=None, vendor=None)

    try:
        text = cpuinfo.read_text()
    except OSError:
        return CpuInfo(model_name=None, vendor=None)

    # Parse the first processor block (up to the first blank line).
    first_block: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, _, value = line.partition(":")
            first_block[key.strip()] = value.strip()

    model_name = first_block.get("model name") or first_block.get("Hardware")
    vendor = first_block.get("vendor_id") or first_block.get("CPU implementer")
    return CpuInfo(model_name=model_name, vendor=vendor)


def _levels_from_topology(topology: MemoryTopologyLike | None) -> tuple[MemoryLevelSignature, ...]:
    """Convert a memory topology to a tuple of deterministic level signatures."""
    if topology is None:
        return ()
    levels: list[MemoryLevelSignature] = []
    for lvl in topology:
        if isinstance(lvl, MemoryLevelInfo):
            levels.append(
                MemoryLevelSignature(
                    name=lvl.name,
                    size_bytes=int(lvl.size.value),
                    instances=lvl.instances,
                    num_sharing_threads=lvl.num_sharing_threads,
                )
            )
    return tuple(levels)


def detect_machine_signature() -> MachineSignature:
    """Standalone detection from /proc/cpuinfo + sysfs.

    No C probes and no execution interface are used, so this is safe to call from
    the ``profile`` subcommand without compiling any detection harness. Falls
    back gracefully:

        - model_name: read from /proc/cpuinfo, or ``platform.machine()``.
        - vendor: read from /proc/cpuinfo, or empty string.
        - memory_levels: built from sysfs via :class:`MemoryTopology` if
          available, else an empty tuple.
        - vector_length: ``None`` (requires a C probe, unavailable standalone).
    """
    cpu_info = read_cpuinfo()
    model_name = cpu_info.model_name or platform.machine()
    vendor = cpu_info.vendor or ""

    try:
        topology: MemoryTopologyLike | None = MemoryTopology()
    except OSError:
        topology = None
    memory_levels = _levels_from_topology(topology)

    return MachineSignature(
        model_name=model_name,
        arch=platform.machine(),
        vendor=vendor,
        memory_levels=memory_levels,
        vector_length=None,
    )


def signature_from_architecture(arch: Architecture) -> MachineSignature:
    """Build a signature from a resolved :class:`Architecture`.

    Used by the ``benchmark`` flow, where the architecture has already been
    detected (including the vector length from a C probe). Falls back to
    ``arch.arch`` then ``"unknown"`` for the model name when unavailable.
    """
    model_name = arch.model_name or arch.arch or "unknown"
    vendor = arch.vendor or ""
    return MachineSignature(
        model_name=model_name,
        arch=arch.arch or platform.machine(),
        vendor=vendor,
        memory_levels=_levels_from_topology(arch.memory_topology),
        vector_length=arch.vector_length,
    )


def _short_model_name(name: str) -> str:
    """Reduce a verbose CPU model string to a short, filesystem-safe token.

    Steps:
        1. Remove parentheticals (e.g. "(R)", "(TM)").
        2. Strip known vendor prefixes (AMD, Intel, ARM, Ampere Computing).
        3. Strip `` with ...`` suffixes (e.g. "with Radeon Graphics").
        4. Collapse whitespace into dashes.
        5. Keep only ``[a-zA-Z0-9-]``.
        6. Collapse repeated dashes, strip leading/trailing dashes.
        7. Truncate to 30 characters (trimming a trailing dash).
        8. Return ``"unknown"`` if the result is empty.

    Examples:
        ``"AMD Ryzen 7 7735HS with Radeon Graphics"`` -> ``"Ryzen-7-7735HS"``
        ``"Intel(R) Core(TM) i7-14700K"`` -> ``"Core-i7-14700K"``
    """
    # 1. Remove parentheticals.
    value = re.sub(r"\([^)]*\)", "", name)
    # 2. Remove vendor prefixes.
    value = re.sub(r"^(AMD|Intel|ARM|Ampere Computing)\s+", "", value, flags=re.IGNORECASE)
    # 3. Remove " with ..." suffix.
    value = re.sub(r"\s+with\s+.*$", "", value, flags=re.IGNORECASE)
    # 4. Collapse whitespace to dashes.
    value = re.sub(r"\s+", "-", value.strip())
    # 5. Keep only alphanumerics and dashes.
    value = re.sub(r"[^a-zA-Z0-9-]", "", value)
    # 6. Collapse multiple dashes, strip leading/trailing dashes.
    value = re.sub(r"-+", "-", value).strip("-")
    # 7. Truncate to 30 chars, trimming a trailing dash.
    value = value[:30].rstrip("-")
    # 8. Empty -> "unknown".
    return value or "unknown"


def generate_run_name(signature: MachineSignature) -> str:
    """Generate ``"<short_model>_<config_hash>"`` (e.g. ``"Ryzen-7-7735HS_59486dd1"``)."""
    return f"{_short_model_name(signature.model_name)}_{signature.config_hash}"
