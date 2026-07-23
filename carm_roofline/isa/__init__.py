from __future__ import annotations

from carm_roofline.isa.arm import ArmNeon, ArmScalar, ArmSVE
from carm_roofline.isa.base import BaseISA
from carm_roofline.isa.riscv import RISCV_RVV, RISCV_RVV_071, RISCVScalar
from carm_roofline.isa.x86 import X86AVX, X86AVX2, X86AVX512, X86SSE, X86Scalar

INCOMPATIBLE_ISAS = {
    frozenset({RISCV_RVV_071, RISCV_RVV}),
}

__all__ = [
    "INCOMPATIBLE_ISAS",
    "RISCV_RVV",
    "RISCV_RVV_071",
    "X86AVX",
    "X86AVX2",
    "X86AVX512",
    "X86SSE",
    "ArmNeon",
    "ArmSVE",
    "ArmScalar",
    "BaseISA",
    "RISCVScalar",
    "X86Scalar",
]
