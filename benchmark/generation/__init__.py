from __future__ import annotations

from .arm import ArmNeon, ArmScalar, ArmSVE
from .code_gen import DataType, Operation
from .isa import BaseISA
from .parameters import ArithmeticBenchmarkParams, BenchmarkParams, MemoryBenchmarkParams, MemoryLayoutMode
from .riscv import RISCV_RVV, RISCV_RVV_071, RISCVScalar
from .x86 import X86AVX, X86AVX2, X86AVX512, X86SSE, X86Scalar

ALL_ISAS: tuple[type[BaseISA], ...] = (
    ArmScalar,
    ArmNeon,
    ArmSVE,
    RISCVScalar,
    RISCV_RVV_071,
    RISCV_RVV,
    X86Scalar,
    X86SSE,
    X86AVX,
    X86AVX2,
    X86AVX512,
)

ISA_NAME_TO_CLASS: dict[str, type[BaseISA]] = {isa.name: isa for isa in ALL_ISAS}

# Special incompatibilities within the same family (e.g., different RVV versions)
INCOMPATIBLE_ISAS = {
    frozenset({RISCV_RVV_071, RISCV_RVV}),  # RVV 0.7.1 and 1.0 are incompatible
}

__all__ = [
    "ALL_ISAS",
    "INCOMPATIBLE_ISAS",
    "ISA_NAME_TO_CLASS",
    "RISCV_RVV",
    "RISCV_RVV_071",
    "X86AVX",
    "X86AVX2",
    "X86AVX512",
    "X86SSE",
    "ArithmeticBenchmarkParams",
    "ArmNeon",
    "ArmSVE",
    "ArmScalar",
    "BaseISA",
    "BenchmarkParams",
    "DataType",
    "MemoryBenchmarkParams",
    "MemoryLayoutMode",
    "Operation",
    "RISCVScalar",
    "X86Scalar",
]
