from __future__ import annotations

import ctypes
import platform
import subprocess
import tempfile
from pathlib import Path

from carm_roofline.exec_interface import ExecutionInterface
from carm_roofline.isa import RISCV_RVV, RISCV_RVV_071, RISCVScalar

from .detect import ROOT, DetectedArchitecture, TestContext, run_generic_tests

# Linux UAPI: __NR_riscv_hwprobe (RISC-V), RISCV_HWPROBE_KEY_IMA_EXT_0,
# RISCV_HWPROBE_EXT_V. The kernel defines this bit specifically as RVV 1.0.
_RISCV_HWPROBE_SYSCALL = 258
_RISCV_HWPROBE_KEY_IMA_EXT_0 = 4
_RISCV_HWPROBE_EXT_V = 1 << 2


class _RiscvHwprobe(ctypes.Structure):
    _fields_ = [("key", ctypes.c_int64), ("value", ctypes.c_uint64)]


def detect(threads: int = 1) -> DetectedArchitecture:
    """Detect RISC-V vector ISA only after the target executes its semantics."""
    exec_iface = _get_execution_interface()
    probe = ROOT / "tests" / "riscv" / "rvv_version.c"
    isa_list = [RISCVScalar.name]
    features: frozenset[str] = frozenset()

    if probe.exists():
        native = exec_iface.is_native()
        hwprobe_v = _linux_hwprobe_has_v() if native else None
        versions: tuple[str, ...] = ("RISCV_RVV",) if hwprobe_v is True else ("RISCV_RVV_0_7_1",)
        if hwprobe_v is None:
            # Without a kernel declaration, only a child process running a
            # version-specific semantic check can distinguish the ISA safely.
            versions = ("RISCV_RVV", "RISCV_RVV_0_7_1")
        for define in versions:
            if _can_run_rvv(exec_iface, probe, define):
                isa_list.append(RISCV_RVV.name if define == "RISCV_RVV" else RISCV_RVV_071.name)
                features = frozenset({"riscv_vector"})
                break

    ctx = TestContext(family="riscv", features=features)
    detected = run_generic_tests(ctx, threads=threads)
    detected.isa = isa_list
    return detected


def _get_execution_interface() -> ExecutionInterface:
    """Lazy import to avoid circular dependency with __init__.py."""
    from . import get_execution_interface

    return get_execution_interface()


def _linux_hwprobe_has_v() -> bool | None:
    """Return Linux's RVV 1.0 hardware claim, or None when hwprobe is unavailable."""
    if platform.system() != "Linux" or not platform.machine().lower().startswith("riscv"):
        return None

    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    pair = _RiscvHwprobe(_RISCV_HWPROBE_KEY_IMA_EXT_0, 0)
    result = syscall(
        _RISCV_HWPROBE_SYSCALL,
        ctypes.byref(pair),
        1,
        0,
        None,
        0,
    )
    if result != 0 or pair.key == -1:
        return None
    return bool(pair.value & _RISCV_HWPROBE_EXT_V)


def _can_run_rvv(exec_iface: ExecutionInterface, source_path: Path, define: str) -> bool:
    """Compile and run a semantic RVV probe in the requested target environment.

    Unsupported instruction dialects are expected during compiler probing, so compiler stderr is captured without 
    warning. The executable is isolated from the caller, so unsupported instructions terminate only the probe process. 
    Successful compilation alone proves nothing about the target CPU or its OS-enabled vector state.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        binary_path = Path(temp_dir) / "rvv_version"
        try:
            result = exec_iface.compile(
                str(source_path),
                str(binary_path),
                f"-D{define}",
                check=False,
                capture_output=True,
                timeout=5,
                warn_on_stderr=False,
            )
            if result.returncode != 0:
                return False
            run = exec_iface.run(
                str(binary_path),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        expected = f'{{"rvv_version":"{define}"}}'
        return run.returncode == 0 and run.stdout.strip() == expected
