"""GPU memory topology: data model and device-property discovery.

Defines the GPU memory hierarchy model (``GPUMemoryLevel``, ``GPUMemoryTopology``)
and ``discover_gpu_memory_topology``, which queries device properties through the
CUDA driver API (NVIDIA) or the KFD sysfs topology with amd-smi/rocminfo
fallback (AMD).

The topology feeds benchmark suite generators (Phase 3+) with cache sizes and
SM counts for working-set sizing.
"""

from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carm_roofline.core import Bandwidth, Bytes, UserError
from carm_roofline.gpu.compute_capability import ComputeCapability
from carm_roofline.gpu.types import GPUVendor
from carm_roofline.output_utils import warn

# CUDA driver API device attribute enums. Values from /usr/include/cuda.h;
# these enum values have been stable since CUDA 5.0.
_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97

_CUDA_SUCCESS = 0

# L2 cache sizes (bytes) per AMD gfx architecture, from AMD architectural
# specifications. Unrecognized architectures fall back to a conservative 4 MiB
# (see _AMD_DEFAULT_L2_BYTES). Expandable in later phases without changing the
# data model. Used only when the KFD sysfs topology is unavailable.
_MB = 1024 * 1024
_AMD_L2_CACHE_BYTES: dict[str, int] = {
    "gfx908": 8 * _MB,  # MI100 (CDNA1)
    "gfx90a": 8 * _MB,  # MI250X (CDNA2)
    "gfx942": 16 * _MB,  # MI300X (CDNA3)
    "gfx1030": 4 * _MB,  # Navi 21 (RDNA2)
    "gfx1035": 2 * _MB,  # Navi 24
    "gfx1100": 6 * _MB,  # Navi 31 (RDNA3)
    "gfx1102": 2 * _MB,  # Navi 33
    "gfx1150": 8 * _MB,  # Navi 48 (RDNA4)
}
_AMD_DEFAULT_L2_BYTES = 4 * _MB
# CDNA2/3 and RDNA3 all expose 64 KiB max shared memory (LDS) per work group.
# Used only when the KFD sysfs topology is unavailable.
_AMD_SHARED_MEMORY_BYTES = 64 * 1024

# AMD KFD sysfs topology root. Present on Linux with the amdgpu KFD driver;
# absent (or without GPU nodes) on other platforms/drivers.
_KFD_TOPOLOGY_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
# Device-local memory heaps (HSA_MEM_HEAP_TYPE_FB_PUBLIC=1, FB_PRIVATE=2).
# heap_type 0 = system memory, 3 = GPU GTT — both excluded from VRAM.
_KFD_DEVICE_HEAP_TYPES = frozenset({1, 2})

# Size unit multipliers for amd-smi "size" objects (MB reported as binary MB).
_SIZE_UNIT_MULTIPLIERS = {
    "B": 1,
    "KB": 1024,
    "KIB": 1024,
    "MB": 1024 * 1024,
    "MIB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
    "GIB": 1024 * 1024 * 1024,
    "TB": 1024**4,
    "TIB": 1024**4,
}


@dataclass(frozen=True)
class GPUMemoryLevel:
    """A single level of the GPU memory hierarchy.

    Attributes:
        name: Human-readable level name (e.g. ``"Shared/L1"``, ``"L2"``, ``"Global"``).
        size: Capacity in bytes.
        sm_count: Total SM (compute unit) count on this device.
        bandwidth: Peak bandwidth, if known (populated from benchmark results in Phase 8).
    """

    name: str
    size: Bytes
    sm_count: int
    bandwidth: Bandwidth | None = None

    def __repr__(self) -> str:
        bw = f", bw={self.bandwidth}" if self.bandwidth is not None else ""
        return f"GPUMemoryLevel({self.name!r}, {self.size}, sm={self.sm_count}{bw})"


@dataclass(frozen=True)
class GPUMemoryTopology:
    """The GPU memory hierarchy for a single device.

    Attributes:
        vendor: GPU vendor (``GPUVendor.NVIDIA`` or ``GPUVendor.AMD``).
        levels: Memory levels ordered from closest to farthest from the SMs.
    """

    vendor: GPUVendor
    levels: tuple[GPUMemoryLevel, ...]

    @property
    def sm_count(self) -> int:
        """Total SM count on this device (device-wide, identical on every level)."""
        return self.levels[0].sm_count if self.levels else 0

    @property
    def shared_l1(self) -> GPUMemoryLevel | None:
        """The shared memory / L1 level, or ``None`` if absent."""
        return next((level for level in self.levels if "l1" in level.name.lower()), None)

    @property
    def l2(self) -> GPUMemoryLevel | None:
        """The L2 cache level, or ``None`` if absent."""
        return next((level for level in self.levels if "l2" in level.name.lower()), None)

    @property
    def global_(self) -> GPUMemoryLevel | None:
        """The global (device) memory level, or ``None`` if absent."""
        return next((level for level in self.levels if "global" in level.name.lower()), None)

    def __repr__(self) -> str:
        levels = ", ".join(str(level) for level in self.levels)
        return f"GPUMemoryTopology(vendor={self.vendor.value}, levels=[{levels}])"


def discover_gpu_memory_topology(
    vendor: GPUVendor,
    device: int = 0,
    compute_capability: ComputeCapability | None = None,
) -> GPUMemoryTopology:
    """Discover the memory topology of the given GPU device.

    NVIDIA devices are queried through the CUDA driver API (``libcuda.so``);
    AMD devices are discovered from the KFD sysfs topology
    (``/sys/class/kfd/kfd/topology/nodes/``) when available, falling back to the
    ``amd-smi``/``rocminfo`` CLI tools plus architecture lookup tables.

    Args:
        vendor: GPU vendor.
        device: GPU device index (vendor-relative, matches Phase 1 detection).
        compute_capability: Optionally pre-detected compute capability (avoids
            re-parsing CLI output for the gfx architecture).

    Returns:
        The discovered memory topology.

    Raises:
        UserError: No usable discovery path for the given vendor/device.
    """
    if vendor == GPUVendor.NVIDIA:
        return _discover_nvidia(device)
    if vendor == GPUVendor.AMD:
        return _discover_amd(device, compute_capability)
    raise UserError(f"Unsupported GPU vendor: {vendor}")


def _discover_nvidia(device: int) -> GPUMemoryTopology:
    """Discover the memory topology of an NVIDIA GPU via the CUDA driver API."""
    try:
        cuda = ctypes.CDLL("libcuda.so")
    except OSError:
        raise UserError("libcuda.so not found: install NVIDIA drivers") from None

    if cuda.cuInit(0) != _CUDA_SUCCESS:
        raise UserError("cuInit failed: CUDA driver API could not be initialized")

    handle = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(handle), device) != _CUDA_SUCCESS:
        raise UserError(f"cuDeviceGet({device}) failed: no CUDA device at index {device}")

    def get_attribute(attr: int) -> int:
        value = ctypes.c_int(0)
        if cuda.cuDeviceGetAttribute(ctypes.byref(value), attr, handle) != _CUDA_SUCCESS:
            raise UserError(f"cuDeviceGetAttribute({attr}) failed for device {device}")
        return value.value

    sm_count = get_attribute(_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    l2_size = get_attribute(_CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

    # Max opt-in shared memory per block; pre-CC 7.0 GPUs report 0 for the
    # opt-in attribute and fall back to the plain per-block value.
    shared = get_attribute(_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    if shared == 0:
        shared = get_attribute(_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    return GPUMemoryTopology(
        vendor=GPUVendor.NVIDIA,
        levels=(
            GPUMemoryLevel(name="Shared/L1", size=Bytes(shared), sm_count=sm_count),
            GPUMemoryLevel(name="L2", size=Bytes(l2_size), sm_count=sm_count),
            GPUMemoryLevel(name="Global", size=Bytes(_nvidia_total_vram(device)), sm_count=sm_count),
        ),
    )


def _nvidia_total_vram(device: int) -> int:
    """Query total global memory (bytes) via nvidia-smi, consistent with Phase 1."""
    if shutil.which("nvidia-smi") is None:
        raise UserError("nvidia-smi not found: cannot query GPU memory size")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "-i", str(device), "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise UserError(f"nvidia-smi failed to query memory.total for device {device}: {e.stderr.strip()}") from e
    line = result.stdout.strip()
    if not line:
        raise UserError(f"nvidia-smi returned empty memory.total data for device {device}")
    try:
        return Bytes.from_string(line).value
    except ValueError:
        raise UserError(f"Could not parse nvidia-smi memory size {line!r}") from None


def _discover_amd(device: int, compute_capability: ComputeCapability | None) -> GPUMemoryTopology:
    """Discover the memory topology of an AMD GPU via the KFD sysfs topology when
    available, falling back to amd-smi/rocminfo + lookup tables."""
    kfd_topology = _discover_amd_from_kfd(device)
    if kfd_topology is not None:
        return kfd_topology

    gfx_arch = compute_capability.gfx_arch if compute_capability is not None else None
    vram_bytes: int | None = None
    compute_units: int | None = None

    if shutil.which("amd-smi") is not None:
        asic_data = _query_amd_smi(device)
        if asic_data is not None:
            if gfx_arch is None:
                gfx_arch = _amd_smi_gfx_arch(asic_data)
            vram_bytes = _amd_vram_bytes(asic_data)
        if vram_bytes is None:
            # Older amd-smi builds put VRAM under the separate -v (--vram) section.
            vram_data = _query_amd_smi(device, flag="v")
            if vram_data is not None:
                vram_bytes = _amd_vram_bytes(vram_data)

    # rocminfo fills any remaining gaps: gfx arch, SM (compute unit) count, and
    # VRAM size ("Max Memory Size" or first pool "Size") when amd-smi is
    # unavailable or does not report VRAM.
    rocm_agent = _rocminfo_gpu_agent(device)
    if rocm_agent is not None:
        rocm_arch, rocm_units, rocm_vram = rocm_agent
        if gfx_arch is None:
            gfx_arch = rocm_arch
        if vram_bytes is None and rocm_vram is not None:
            vram_bytes = rocm_vram
        if compute_units is None:
            compute_units = rocm_units

    if gfx_arch is None:
        raise UserError(f"Cannot determine the gfx architecture of AMD GPU {device} (amd-smi and rocminfo unavailable)")
    if vram_bytes is None:
        raise UserError(f"Cannot determine the VRAM size of AMD GPU {device} (amd-smi and rocminfo unavailable)")
    if compute_units is None:
        compute_units = 1  # rocminfo always reports Compute Unit; defensive fallback

    l2_bytes = _AMD_L2_CACHE_BYTES.get(gfx_arch)
    if l2_bytes is None:
        warn(f"Unknown AMD gfx architecture {gfx_arch}; assuming 4 MiB L2 cache")
        l2_bytes = _AMD_DEFAULT_L2_BYTES

    return GPUMemoryTopology(
        vendor=GPUVendor.AMD,
        levels=(
            GPUMemoryLevel(name="Shared/L1", size=Bytes(_AMD_SHARED_MEMORY_BYTES), sm_count=compute_units),
            GPUMemoryLevel(name="L2", size=Bytes(l2_bytes), sm_count=compute_units),
            GPUMemoryLevel(name="Global", size=Bytes(vram_bytes), sm_count=compute_units),
        ),
    )


def _query_amd_smi(device: int, flag: str = "a") -> dict[str, Any] | None:
    """Run ``amd-smi static -g<device> -<flag> --json`` and parse the top-level object.

    ``-a`` (--asic) is the Phase 1-compatible query; some builds report VRAM only
    under ``-v`` (--vram).
    """
    try:
        result = subprocess.run(
            ["amd-smi", "static", f"-g{device}", f"-{flag}", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _amd_vram_bytes(data: dict[str, Any]) -> int | None:
    """Extract total VRAM in bytes from amd-smi static JSON.

    Prefers the structured ``vram.size`` object (``{"value": <MB>, "unit": "MB"}``)
    reported by the CLI, falling back to the library-style ``asic.vram_size``
    field (raw bytes).
    """
    vram = data.get("vram")
    if isinstance(vram, dict):
        size = vram.get("size")
        if isinstance(size, dict) and size.get("value") is not None:
            value = _parse_bytes_value(size.get("value"))
            unit = str(size.get("unit", "MB")).upper().replace(" ", "")
            if value is not None:
                return int(value * _size_unit_multiplier(unit))
    asic = data.get("asic")
    if isinstance(asic, dict) and asic.get("vram_size") is not None:
        return _parse_bytes_value(asic.get("vram_size"))
    return None


def _amd_smi_gfx_arch(data: dict[str, Any]) -> str | None:
    """Extract the gfx architecture from amd-smi static JSON, if reported."""
    asic = data.get("asic")
    if not isinstance(asic, dict):
        return None
    arch = asic.get("target_graphics_version")
    if isinstance(arch, str) and arch and arch != "N/A":
        return arch
    return None


def _parse_bytes_value(value: object) -> int | None:
    """Parse a raw byte count given as int, float, or string."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return Bytes.from_string(s).value
        except ValueError:
            return None
    return None


def _size_unit_multiplier(unit: str) -> int:
    """Map a size unit string (``"MB"``, ``"GiB"``, ...) to its byte multiplier."""
    return _SIZE_UNIT_MULTIPLIERS.get(unit, 1024 * 1024)


def _rocminfo_gpu_agent(device: int) -> tuple[str, int, int | None] | None:
    """Query rocminfo for the ``device``-th GPU agent.

    Returns ``(gfx_arch, compute_units, vram_bytes)`` where ``vram_bytes`` is the
    agent's ``Max Memory Size`` if reported, else the first ``Pool Info`` pool
    size (rocminfo reports pool sizes in KB), or ``None`` if rocminfo fails or
    the index is out of range.
    """
    try:
        result = subprocess.run(["rocminfo"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    lines = result.stdout.splitlines()
    n = len(lines)
    gpu_count = -1  # zero-based GPU agent index
    i = 0
    while i < n:
        line = lines[i].strip()
        # Agent delimiter: "*******" surrounding "Agent N"
        if (
            line == "*******"
            and i + 2 < n
            and lines[i + 1].strip().startswith("Agent")
            and lines[i + 2].strip() == "*******"
        ):
            found_name: str | None = None
            found_dev_type: str | None = None
            compute_units: int | None = None
            max_memory: int | None = None
            pool_size: int | None = None
            in_pool_info = False
            j = i + 3
            while j < n and lines[j].strip() != "*******":
                sj = lines[j].strip()
                if sj.startswith("Name:") and found_name is None:
                    found_name = sj.split(":", 1)[1].strip()
                elif "Device Type:" in sj:
                    found_dev_type = sj.split(":", 1)[1].strip()
                elif sj.startswith("Compute Unit:") and compute_units is None:
                    compute_units = _parse_int_field(sj)
                elif sj.startswith("Max Memory Size:") and max_memory is None:
                    max_memory = _parse_int_field(sj)
                elif sj.startswith("Pool Info:"):
                    in_pool_info = True
                elif in_pool_info and sj.startswith("Size:") and pool_size is None:
                    pool_size = _parse_rocminfo_size(sj)
                j += 1
            if found_name and found_dev_type == "GPU":
                gpu_count += 1
                if gpu_count == device:
                    vram = max_memory if max_memory is not None else pool_size
                    return found_name, compute_units if compute_units is not None else 1, vram
            i = j  # skip past this agent block
            continue
        i += 1
    return None


def _parse_kfd_properties(path: Path) -> dict[str, int]:
    """Parse a KFD sysfs ``properties`` file ("key value" lines) into ints.

    Non-integer values and unreadable files yield an empty dict (callers treat
    absent keys as missing fields and fall back to the amd-smi/rocminfo chain).
    """
    try:
        text = path.read_text()
    except OSError:
        return {}
    props: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        try:
            props[parts[0]] = int(parts[1])
        except (ValueError, IndexError):
            continue
    return props


def _discover_amd_from_kfd(device: int) -> GPUMemoryTopology | None:
    """Discover AMD topology from the KFD sysfs topology, or None if unavailable.

    Returns a complete topology only when every field is present (all-or-nothing):
    any missing/partial data falls back to the amd-smi/rocminfo chain.
    """
    root = _KFD_TOPOLOGY_ROOT
    if not root.is_dir():
        return None

    try:
        node_dirs = sorted(root.iterdir())
    except OSError:
        return None

    # GPU nodes have simd_count > 0; the CPU node (node 0) reports 0. location_id
    # is the BDF-packed PCI address ((domain << 16) | (bus << 8) | (device << 3) |
    # function), so numeric sort == Phase 1 PCI-bus order for AMD GPUs.
    candidates: list[tuple[Path, dict[str, int], int]] = []
    for node_dir in node_dirs:
        props = _parse_kfd_properties(node_dir / "properties")
        if props.get("simd_count", 0) > 0:
            try:
                node_index = int(node_dir.name)
            except ValueError:
                node_index = 0
            candidates.append((node_dir, props, node_index))
    candidates.sort(key=lambda c: (c[1].get("location_id") is None, c[1].get("location_id") or 0, c[2]))

    # Guard device < 0 explicitly: bare indexing would wrap for negative indices.
    if not 0 <= device < len(candidates):
        return None

    node_dir, props, _node_index = candidates[device]

    sm_count = props.get("simd_count") or 0
    simd_per_cu = props.get("simd_per_cu") or 0
    if simd_per_cu > 0:
        sm_count //= simd_per_cu

    shared_kb = props.get("lds_size_in_kb")
    shared_bytes = shared_kb * 1024 if shared_kb is not None else None

    l2_bytes = 0
    try:
        cache_dirs = sorted((node_dir / "caches").iterdir())
    except OSError:
        cache_dirs = []
    for cache_dir in cache_dirs:
        cache_props = _parse_kfd_properties(cache_dir / "properties")
        if cache_props.get("level") == 2:
            size_kb = cache_props.get("size")
            if size_kb is not None:
                l2_bytes += size_kb * 1024

    vram_bytes = 0
    try:
        bank_dirs = sorted((node_dir / "mem_banks").iterdir())
    except OSError:
        bank_dirs = []
    for bank_dir in bank_dirs:
        bank_props = _parse_kfd_properties(bank_dir / "properties")
        if bank_props.get("heap_type") in _KFD_DEVICE_HEAP_TYPES:
            size_bytes = bank_props.get("size_in_bytes")
            if size_bytes is not None:
                vram_bytes += size_bytes

    # Any missing/zero field drops the whole KFD result so the amd-smi/rocminfo chain decides.
    if sm_count == 0 or shared_bytes is None or l2_bytes <= 0 or vram_bytes <= 0:
        return None

    return GPUMemoryTopology(
        vendor=GPUVendor.AMD,
        levels=(
            GPUMemoryLevel(name="Shared/L1", size=Bytes(shared_bytes), sm_count=sm_count),
            GPUMemoryLevel(name="L2", size=Bytes(l2_bytes), sm_count=sm_count),
            GPUMemoryLevel(name="Global", size=Bytes(vram_bytes), sm_count=sm_count),
        ),
    )


def _parse_int_field(line: str) -> int | None:
    """Extract the first integer from a ``"Field: value"`` line (e.g. rocminfo)."""
    try:
        return int(line.split(":", 1)[1].strip().split()[0])
    except (ValueError, IndexError):
        return None


def _parse_rocminfo_size(line: str) -> int | None:
    """Parse a rocminfo pool ``Size:`` line like ``Size: 7795976(0x76f508) KB`` into bytes."""
    s = line.split(":", 1)[1].strip() if ":" in line else line
    match = re.match(r"^(\d+)", s)
    if match is None:
        return None
    unit = "B"
    for candidate in ("KB", "MB", "GB"):
        if candidate in s:
            unit = candidate
            break
    return int(match.group(1)) * _size_unit_multiplier(unit)
