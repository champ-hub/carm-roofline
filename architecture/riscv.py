from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark import generation as bench_gen
from exec_interface import ExecutionInterface

from .detect import ROOT, DetectedArchitecture, TestContext, run_generic_tests


def detect(threads: int = 1) -> DetectedArchitecture:
    """Detect RISC-V ISA features and VLEN.

    Returns a DetectedArchitecture with:
        - isa: list of ISA strings (e.g., ["riscv", "riscv_rvv"])
        - vector_length: detected VLEN in bytes (if available)

    Args:
        threads: Number of threads to use for frequency detection
    """
    # Run generic tests (features, cache, vlen, frequency)
    ctx = TestContext(family="riscv")
    detected = run_generic_tests(ctx, threads=threads)

    # Build ISA list with scalar as base
    isa_list = [bench_gen.RISCVScalar.name]

    # Detect RVV version using RISC-V specific test
    exec_iface = _get_execution_interface()
    rvv_version_probe_path = ROOT / "tests" / "riscv" / "rvv_version.c"

    if rvv_version_probe_path.exists():
        # Try compiling with RVV 1.0 flag first (most common)
        if _can_compile_rvv(exec_iface, rvv_version_probe_path, "RISCV_RVV"):
            isa_list.append(bench_gen.RISCV_RVV.name)
        elif _can_compile_rvv(exec_iface, rvv_version_probe_path, "RISCV_RVV_0_7_1"):
            isa_list.append(bench_gen.RISCV_RVV_071.name)

    detected.isa = isa_list

    return detected


def _get_execution_interface() -> ExecutionInterface:
    """Lazy import to avoid circular dependency with __init__.py."""
    from . import get_execution_interface

    return get_execution_interface()


def _can_compile_rvv(exec_iface: ExecutionInterface, source_path: Path, define: str) -> bool:
    """Attempt to compile the RVV version probe with specific flags.

    Args:
        exec_iface: ExecutionInterface instance
        source_path: Path to rvv_version.c
        define: Preprocessor define to set (RISCV_RVV or RISCV_RVV_0_7_1)

    Returns:
        True if compilation succeeds, False otherwise
    """
    with tempfile.NamedTemporaryFile(suffix=".o", delete=True) as tmp:
        result = exec_iface.compile(
            str(source_path),
            tmp.name,
            f"-D{define}",
            "-c",  # Compile only, don't link
            check=False,  # Do not raise on failure, one of tests is expected to fail
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
