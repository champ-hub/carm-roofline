from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from collections import defaultdict

from carm_roofline.core import UserError
from carm_roofline.gpu.compute_capability import ComputeCapability
from carm_roofline.gpu.types import GPUVendor
from carm_roofline.output_utils import warn

# PCI vendor IDs
_NVIDIA_VENDOR = "0x10de"
_AMD_VENDORS = frozenset({"0x1002", "0x1022"})
# PCI class codes: 0x0300=VGA, 0x0302=3D controller, 0x0380=display controller
_GPU_CLASSES = frozenset({"0x0300", "0x0302", "0x0380"})
_PCI_DEVICES = pathlib.Path("/sys/bus/pci/devices")


def _vendor_from_pci_id(vendor_id: str) -> GPUVendor | None:
    """Map a PCI vendor ID string to a ``GPUVendor``."""
    if vendor_id == _NVIDIA_VENDOR:
        return GPUVendor.NVIDIA
    if vendor_id in _AMD_VENDORS:
        return GPUVendor.AMD
    return None


def _enumerate_gpus() -> list[tuple[GPUVendor, int]]:
    """Enumerate GPUs in PCI bus order via sysfs.

    Reads ``/sys/bus/pci/devices/`` to discover all supported GPUs
    sorted by PCI bus address (which matches physical slot order).

    Returns:
        List of ``(vendor, vendor_relative_index)`` tuples in PCI bus
        order. Returns an empty list if sysfs is unavailable or no
        supported GPUs are found.
    """
    if not _PCI_DEVICES.is_dir():
        return []

    gpus: list[tuple[GPUVendor, int]] = []
    vendor_counts: dict[GPUVendor, int] = defaultdict(int)

    for dev_dir in sorted(_PCI_DEVICES.iterdir(), key=lambda p: p.name):
        try:
            class_id = (dev_dir / "class").read_text().strip()
            vendor_id = (dev_dir / "vendor").read_text().strip()
        except (OSError, PermissionError):
            continue

        # Use the first 6 hex digits of the class code
        pci_class = class_id[:6] if len(class_id) >= 6 else class_id
        if pci_class not in _GPU_CLASSES:
            continue

        vendor = _vendor_from_pci_id(vendor_id)
        if vendor is None:
            continue

        rel_idx = vendor_counts[vendor]
        vendor_counts[vendor] += 1
        gpus.append((vendor, rel_idx))

    return gpus


def _detect_nvidia(device: int) -> tuple[GPUVendor, ComputeCapability, str] | None:
    """Detect an NVIDIA GPU at the given device index.

    Returns ``None`` if nvidia-smi is not on PATH, or if no NVIDIA GPU is found
    at the given device index (the device may be an AMD GPU or the index may
    be out of range).
    """
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap,gpu_name", "-i", str(device), "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Not an NVIDIA GPU at this index — allow caller to try AMD
        return None

    line = result.stdout.strip()
    if not line:
        return None  # No NVIDIA GPU at this index

    # Expected format: "8.9, NVIDIA GeForce RTX 4090"
    parts = line.split(", ", 1)
    cc_str = parts[0].strip()
    model_name = parts[1].strip() if len(parts) > 1 else f"GPU {device}"

    cc = ComputeCapability.from_string(cc_str, GPUVendor.NVIDIA)
    return GPUVendor.NVIDIA, cc, model_name


def _get_amd_gfx_arch(device: int) -> str | None:
    """Get the gfx architecture string for the Nth AMD GPU via rocminfo.

    Counts only HSA GPU agents (Device Type: GPU) in ``rocminfo`` output,
    returning the ``Name`` field (e.g. ``"gfx1035"``) of the ``device``-th
    one. Returns ``None`` if rocminfo is unavailable or the index is out
    of range.
    """
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    lines = result.stdout.splitlines()
    i, n = 0, len(lines)
    gpu_count = -1  # zero-based

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
            j = i + 3
            while j < n and lines[j].strip() != "*******":
                sj = lines[j].strip()
                if sj.startswith("Name:") and found_name is None:
                    found_name = sj.split(":", 1)[1].strip()
                elif "Device Type:" in sj:
                    found_dev_type = sj.split(":", 1)[1].strip()
                j += 1
            if found_name and found_dev_type == "GPU":
                gpu_count += 1
                if gpu_count == device:
                    return found_name
            i = j  # skip past this agent block
            continue
        i += 1
    return None


def _detect_amd(device: int) -> tuple[GPUVendor, ComputeCapability, str] | None:
    """Detect an AMD GPU at the given device index.

    Returns ``None`` if amd-smi is not on PATH, or if no AMD GPU is found
    at the given device index (the device may be an NVIDIA GPU or the index
    may be out of range).
    """
    if shutil.which("amd-smi") is None:
        return None

    try:
        result = subprocess.run(
            ["amd-smi", "static", f"-g{device}", "-a", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Not an AMD GPU at this index — allow caller to try NVIDIA
        return None

    try:
        data = json.loads(result.stdout)
        market_name = data.get("asic", {}).get("market_name", f"AMD GPU {device}")
    except (json.JSONDecodeError, AttributeError):
        return None

    # Get gfx arch: try amd-smi target_graphics_version first (newer versions),
    # then fall back to rocminfo
    try:
        arch = data.get("asic", {}).get("target_graphics_version")
    except (KeyError, TypeError):
        arch = None
    gfx_arch = arch if arch and arch != "N/A" else _get_amd_gfx_arch(device)
    if gfx_arch is None:
        return None
    cc = ComputeCapability.from_string(gfx_arch, GPUVendor.AMD)
    return GPUVendor.AMD, cc, market_name


def detect_gpu(device: int = 0) -> tuple[GPUVendor, ComputeCapability, str]:
    """Detect a GPU by its PCI bus index (physical slot order).

    The ``device`` index refers to the Nth GPU in PCI bus order (``--gpu-device 0`` = first physical GPU,
    ``--gpu-device 1`` = second, etc.), regardless of vendor. On a mixed-vendor system the vendor is determined by
    reading PCI vendor IDs from sysfs.

    Falls back to probing NVIDIA first, then AMD, if PCI enumeration via sysfs is unavailable.

    Args:
        device: Zero-based GPU index in PCI bus order.

    Returns:
        Tuple of ``(vendor, compute_capability, model_name)``.

    Raises:
        FileNotFoundError: No GPU found at the given index, or no detection tool is available.
    """
    gpu_list = _enumerate_gpus()
    if gpu_list:
        if device >= len(gpu_list):
            raise FileNotFoundError(f"GPU device {device} not found. Only {len(gpu_list)} GPU(s) detected via sysfs.")
        vendor, rel_idx = gpu_list[device]

        result = _detect_nvidia(rel_idx) if vendor == GPUVendor.NVIDIA else _detect_amd(rel_idx)

        if result is not None:
            return result

        raise FileNotFoundError(
            f"Detected {vendor.value} GPU at device {device} but the corresponding SMI tool is unavailable or failed."
        )

    # Fallback: sysfs unavailable — try each vendor
    result = _detect_nvidia(device)
    if result is not None:
        return result

    result = _detect_amd(device)
    if result is not None:
        return result

    raise FileNotFoundError("No GPU detected. Install NVIDIA drivers (nvidia-smi) or AMD ROCm (amd-smi).")


def detect_gpu_vendor() -> GPUVendor | None:
    """Detect the vendor of the first available GPU.

    Uses sysfs enumeration to identify the first GPU in PCI bus order.
    Falls back to ``shutil.which`` checks for nvidia-smi/amd-smi when
    sysfs is unavailable.

    Returns:
        ``GPUVendor.NVIDIA``, ``GPUVendor.AMD``, or ``None`` if no
        GPU or detection tool is found.
    """
    gpu_list = _enumerate_gpus()
    if gpu_list:
        return gpu_list[0][0]

    if shutil.which("nvidia-smi") is not None:
        return GPUVendor.NVIDIA
    if shutil.which("amd-smi") is not None:
        return GPUVendor.AMD
    return None


def detect_compute_capability(device: int = 0) -> ComputeCapability | None:
    """Detect compute capability, returning ``None`` on failure.

    Args:
        device: Zero-based GPU index in PCI bus order.

    See :func:`detect_gpu` for semantics.
    """
    try:
        _, cc, _ = detect_gpu(device=device)
        return cc
    except (FileNotFoundError, UserError):
        return None


def read_gpu_frequencies(device: int = 0) -> dict[str, int]:
    """Read current SM and memory clock frequencies from an NVIDIA GPU.

    Returns:
        Dictionary with keys ``"sm"`` and ``"mem"``, values in MHz.

    Raises:
        UserError: nvidia-smi is unavailable or fails.
    """
    if shutil.which("nvidia-smi") is None:
        raise UserError("nvidia-smi not found — cannot read GPU frequencies")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "-i", str(device), "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise UserError(f"nvidia-smi failed to read frequencies for device {device}: {e.stderr.strip()}") from e

    line = result.stdout.strip()
    if not line:
        raise UserError(f"nvidia-smi returned empty frequency data for device {device}")

    parts = line.split(", ")
    sm_clock = int(parts[0].strip().split()[0])
    mem_clock = int(parts[1].strip().split()[0])
    return {"sm": sm_clock, "mem": mem_clock}


def _make_smi_args(device: int, *flags: str) -> list[str]:
    """Build nvidia-smi argument list for clock control."""
    args = ["nvidia-smi", "-i", str(device)]
    args.extend(flags)
    return args


def lock_gpu_frequencies(
    device: int = 0,
    sm_clock: int | None = None,
    mem_clock: int | None = None,
) -> None:
    """Lock SM and/or memory clock frequencies on an NVIDIA GPU.

    Args:
        device: GPU device index.
        sm_clock: SM clock frequency in MHz (passed to ``-lgc``).
        mem_clock: Memory clock frequency in MHz (passed to ``-lmc``).

    Raises:
        UserError: nvidia-smi is unavailable or fails.
    """
    if shutil.which("nvidia-smi") is None:
        raise UserError("nvidia-smi not found — cannot lock GPU frequencies")

    try:
        if sm_clock is not None:
            subprocess.run(
                _make_smi_args(device, "-lgc", str(sm_clock)),
                capture_output=True,
                text=True,
                check=True,
            )
        if mem_clock is not None:
            subprocess.run(
                _make_smi_args(device, "-lmc", str(mem_clock)),
                capture_output=True,
                text=True,
                check=True,
            )
    except subprocess.CalledProcessError as e:
        raise UserError(f"nvidia-smi failed to lock frequencies for device {device}: {e.stderr.strip()}") from e


def reset_gpu_clocks(device: int = 0) -> None:
    """Reset GPU clocks to default. Best-effort — warns on failure, never raises.

    Args:
        device: GPU device index.
    """
    if shutil.which("nvidia-smi") is None:
        return

    for flag in ("-rgc", "-rmc"):
        try:
            subprocess.run(
                _make_smi_args(device, flag),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            warn(f"nvidia-smi {flag} failed for device {device}", stacklevel=2)
