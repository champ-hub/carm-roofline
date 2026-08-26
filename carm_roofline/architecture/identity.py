"""Machine identity: deterministic hardware signature and run-name generation.

Provides a shared home for both the ``benchmark`` and ``profile`` subcommands to
derive a human-readable, deterministic run name from non-measured hardware
properties (CPU model, architecture, vendor, memory topology, vector length).

Measured values such as frequencies are deliberately excluded so that the same
machine yields the same name across runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .memory import MemoryLevelInfo, MemoryTopology, MemoryTopologyLike

if TYPE_CHECKING:
    from .architecture import Architecture

_BYTES_PER_GIB = 1024**3
_ARM_IMPLEMENTERS = {
    0x41: "ARM",
}
_ARM_CORE_PARTS = {
    (0x41, 0xD4F): "Neoverse V2",
}


def _parse_hex_identifier(value: str | None) -> int | None:
    """Parse an ARM hexadecimal identifier."""
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return None


def _resolve_cpu_identity(fields: dict[str, str]) -> tuple[str | None, str | None]:
    """Resolve model and vendor names from parsed CPU information."""
    model_name = fields.get("model name") or fields.get("Hardware")
    vendor = fields.get("vendor_id")
    implementer_raw = fields.get("CPU implementer")
    implementer = _parse_hex_identifier(implementer_raw)
    normalized_vendor = (
        _ARM_IMPLEMENTERS[implementer]
        if implementer is not None and implementer in _ARM_IMPLEMENTERS
        else implementer_raw
    )

    if vendor is None:
        vendor = normalized_vendor
    if model_name is None:
        part = _parse_hex_identifier(fields.get("CPU part"))
        core_name = _ARM_CORE_PARTS.get((implementer, part)) if implementer is not None and part is not None else None
        if core_name is not None and normalized_vendor is not None:
            model_name = f"{normalized_vendor} {core_name}"

    return model_name, vendor


@dataclass(frozen=True)
class CpuInfo:
    """CPU identification read from /proc/cpuinfo."""

    model_name: str | None
    vendor: str | None
    family: str | None = None
    model: str | None = None
    stepping: str | None = None
    implementer: str | None = None
    part: str | None = None
    variant: str | None = None
    revision: str | None = None


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

    def _canonical_hash_string(self) -> str:
        """Canonical string fed to SHA-256 for the config hash.

        Format: ``arch=<arch>;vendor=<vendor>;levels=<levels>`` where
        ``<levels>`` is a ``|``-separated list of
        ``<name>:<size_bytes>:<instances>:<num_sharing_threads>``.
        """
        levels_str = "|".join(
            f"{lvl.name}:{lvl.size_bytes}:{lvl.instances}:{lvl.num_sharing_threads}" for lvl in self.memory_levels
        )
        return f"arch={self.arch};vendor={self.vendor};levels={levels_str}"

    @property
    def config_hash(self) -> str:
        """Deterministic 8-char hex SHA256 of non-model-name fields.

        Hash inputs: arch, vendor, memory_levels (name, size, instances, sharing).

        .. note::
            DRAM ``size_bytes`` comes from ``/proc/zoneinfo`` ``present`` pages
            (physical capacity, stable across reboots).  Falls back to a
            rounded-``MemTotal`` value when zoneinfo is unavailable.
            ``num_sharing_threads`` is set to 0 for DRAM (not a hardware
            property; CPU online count may vary).  Cache-level fields are
            used verbatim.
        """
        return hashlib.sha256(self._canonical_hash_string().encode("utf-8")).hexdigest()[:8]

    def to_dict(self) -> dict[str, object]:
        """Serialize all fields for machine.json debugging output."""
        return {
            "model_name": self.model_name,
            "arch": self.arch,
            "vendor": self.vendor,
            "config_hash": self.config_hash,
            "hash_input": self._canonical_hash_string(),
            "memory_levels": [
                {
                    "name": lvl.name,
                    "size_bytes": lvl.size_bytes,
                    "instances": lvl.instances,
                    "num_sharing_threads": lvl.num_sharing_threads,
                }
                for lvl in self.memory_levels
            ],
        }


def read_cpuinfo() -> CpuInfo:
    """Read CPU identification fields from /proc/cpuinfo.

    Parses the first processor block. Returns ``CpuInfo(None, None)`` if
    /proc/cpuinfo is unavailable or cannot be parsed.

    Recognized keys:
        - x86: "model name" (model), "vendor_id" (vendor), "cpu family" (family),
          "model" (model), "stepping" (stepping)
        - ARM 32-bit: "Hardware" (fallback model name)
        - ARM 64-bit: "CPU implementer", "CPU part", "CPU variant", and
          "CPU revision"
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

    model_name, vendor = _resolve_cpu_identity(first_block)
    return CpuInfo(
        model_name=model_name,
        vendor=vendor,
        family=first_block.get("cpu family"),
        model=first_block.get("model"),
        stepping=first_block.get("stepping"),
        implementer=first_block.get("CPU implementer"),
        part=first_block.get("CPU part"),
        variant=first_block.get("CPU variant"),
        revision=first_block.get("CPU revision"),
    )


def _get_physical_ram_bytes() -> int | None:
    """Total physical RAM (bytes) from ``/proc/zoneinfo``, or ``None``.

    Sums the ``present`` field across all zones — this is the number of
    physical pages present, derived from the firmware's e820/device-tree
    memory map at boot.  It reflects hardware capacity and is stable across
    reboots, unlike ``MemTotal`` which subtracts kernel runtime reservations.
    """
    try:
        with open("/proc/zoneinfo") as f:
            total_present = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == "present":
                    total_present += int(parts[1])
        return total_present * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def _levels_from_topology(topology: MemoryTopologyLike | None) -> tuple[MemoryLevelSignature, ...]:
    """Convert a memory topology to a tuple of deterministic level signatures.

    DRAM levels use the ``present`` page count from ``/proc/zoneinfo`` (a
    hardware-capacity value stable across reboots).  Falls back to rounding
    the topology's ``MemTotal``-based size to the nearest GiB when zoneinfo
    is unavailable (containers, odd kernels).

    ``num_sharing_threads`` is set to 0 for DRAM (not a hardware property;
    ``shared_cpu_list`` varies with CPU online count).

    Cache levels (L1/L2/L3) are passed through unchanged.
    """
    # Read physical RAM once — stable firmware value
    physical_ram = _get_physical_ram_bytes()

    if topology is None:
        return ()
    levels: list[MemoryLevelSignature] = []
    for lvl in topology:
        if isinstance(lvl, MemoryLevelInfo):
            if lvl.name == "DRAM":
                if physical_ram is not None:
                    size_bytes = physical_ram
                else:
                    size_bytes = round(int(lvl.size) / _BYTES_PER_GIB) * _BYTES_PER_GIB
                levels.append(
                    MemoryLevelSignature(
                        name=lvl.name,
                        size_bytes=size_bytes,
                        instances=lvl.instances,
                        num_sharing_threads=0,
                    )
                )
            else:
                levels.append(
                    MemoryLevelSignature(
                        name=lvl.name,
                        size_bytes=int(lvl.size),
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
    )


def signature_from_architecture(arch: Architecture) -> MachineSignature:
    """Build a signature from a resolved :class:`Architecture`.

    Used by the ``benchmark`` flow, where the architecture has already been
    detected (including C-probed fields such as vector length).
    Falls back to ``arch.arch`` then ``"unknown"`` for the model name when
    unavailable.
    """
    model_name = arch.model_name or arch.arch or "unknown"
    vendor = arch.vendor or ""
    return MachineSignature(
        model_name=model_name,
        arch=arch.arch or platform.machine(),
        vendor=vendor,
        memory_levels=_levels_from_topology(arch.memory_topology),
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


def write_machine_json(signature: MachineSignature, directory: Path) -> None:
    """Write machine.json to *directory* if it does not already exist.

    The file contains the full MachineSignature serialized as JSON, including
    all hash inputs and the computed hash, so future runs can debug why a
    machine hash changed.
    """
    path = directory / "machine.json"
    if path.exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(signature.to_dict(), f, indent=2, sort_keys=True)
