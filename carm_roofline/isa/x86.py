from __future__ import annotations

from typing import Any

from carm_roofline.benchmark.generation.code_gen import ControlInstructions, TypedInstructions, instruction as inst
from carm_roofline.benchmark.generation.code_gen.register import CyclicRegisterSet, HelperRegisterSet, TypedRegisterSets
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation
from carm_roofline.isa.base import BaseISA


class BaseX86(BaseISA):
    name = "x86"
    family = "x86"
    # This is very large in practice (up to 32-bits relative, with more efficient encoding up to 8-bits)
    # A reasonably large value is picked arbitrarily here (should fit most icaches)
    max_branch_insts = 1024
    # Again, very large in practice. Arbitrary offset is chosen here to keep it under 16-bits
    max_mem_offset_bytes = 2**15
    # Allows for any operands that can be represented as 32-bit two's complement
    max_immediate = 2**31

    # Fits in a Gracemont uop cache (512 entries)
    instruction_limit = 512

    class ControlInsts(ControlInstructions):
        load_imm = inst.LoadImm("movq ${imm}, {dst}")
        load_word = inst.LoadWord("movq {ptr}, {dst}")
        branch_nz = inst.BranchNotZero("jnz {tgt}")
        add_imm = inst.AddImm("addq ${imm}, {src}")
        sub_imm = inst.SubImm("subq ${imm}, {src}")
        add = inst.Add("addq {src2}, {dst}")

    control_instructions = ControlInsts()
    helper_registers = HelperRegisterSet("%{}", ("rax", "rdi", "r8", "r9", "r10"))

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class X86Scalar(BaseX86):
    __registers_float = CyclicRegisterSet("%xmm{}", [(0, 15)])

    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "vaddss {}, {}, {}",
                ArithmeticOperation.mul: "vmulss {}, {}, {}",
                ArithmeticOperation.div: "vdivss {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132ss {}, {}, {}",
                MemoryOperation.ld: "movss {off}({ptr}), {reg}",
                MemoryOperation.st: "movss {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "vaddsd {}, {}, {}",
                ArithmeticOperation.mul: "vmulsd {}, {}, {}",
                ArithmeticOperation.div: "vdivsd {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132sd {}, {}, {}",
                MemoryOperation.ld: "movsd {off}({ptr}), {reg}",
                MemoryOperation.st: "movsd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: CyclicRegisterSet("%xmm{}", [(0, 15)]),
                DataType.f64: CyclicRegisterSet("%xmm{}", [(0, 15)]),
            }
        )


class X86SSE(X86Scalar):
    name = "x86_sse"
    BITS = 128

    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "addps {}, {}",
                ArithmeticOperation.mul: "mulps {}, {}",
                ArithmeticOperation.div: "divps {}, {}",
                ArithmeticOperation.fma: "vfmadd132ps {}, {}, {}",
                MemoryOperation.ld: "movaps {off}({ptr}), {reg}",
                MemoryOperation.st: "movaps {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "addpd {}, {}",
                ArithmeticOperation.mul: "mulpd {}, {}, {}",
                ArithmeticOperation.div: "divpd {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132pd {}, {}, {}",
                MemoryOperation.ld: "movapd {off}({ptr}), {reg}",
                MemoryOperation.st: "movapd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def bytes_per_inst(self, data_type: DataType) -> int:
        return self.BITS // 8

    def ops_per_inst(self, data_type: DataType, op: ArithmeticOperation) -> int:
        return op.ops() * self.BITS // data_type.bits()


class X86AVX(X86SSE):
    name = "x86_avx"
    BITS = 256

    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "vaddps {}, {}, {}",
                ArithmeticOperation.mul: "vmulps {}, {}, {}",
                ArithmeticOperation.div: "vdivps {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132ps {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "vaddpd {}, {}, {}",
                ArithmeticOperation.mul: "vmulpd {}, {}, {}",
                ArithmeticOperation.div: "vdivpd {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132pd {}, {}, {}",
                MemoryOperation.ld: "vmovapd {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovapd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: CyclicRegisterSet("%ymm{}", [(0, 15)]),
                DataType.f64: CyclicRegisterSet("%ymm{}", [(0, 15)]),
            }
        )


class X86AVX2(X86AVX):
    name = "x86_avx2"
    BITS = 256

    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "vaddps {}, {}, {}",
                ArithmeticOperation.mul: "vmulps {}, {}, {}",
                ArithmeticOperation.div: "vdivps {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132ps {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "vaddpd {}, {}, {}",
                ArithmeticOperation.mul: "vmulpd {}, {}, {}",
                ArithmeticOperation.div: "vdivpd {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132pd {}, {}, {}",
                MemoryOperation.ld: "vmovapd {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovapd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class X86AVX512(X86SSE):
    name = "x86_avx512"
    BITS = 512

    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "vaddps {}, {}, {}",
                ArithmeticOperation.mul: "vmulps {}, {}, {}",
                ArithmeticOperation.div: "vdivps {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132ps {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "vaddpd {}, {}, {}",
                ArithmeticOperation.mul: "vmulpd {}, {}, {}",
                ArithmeticOperation.div: "vdivpd {}, {}, {}",
                ArithmeticOperation.fma: "vfmadd132pd {}, {}, {}",
                MemoryOperation.ld: "vmovapd {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovapd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: CyclicRegisterSet("%zmm{}", [(0, 31)]),
                DataType.f64: CyclicRegisterSet("%zmm{}", [(0, 31)]),
            }
        )
