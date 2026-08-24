from __future__ import annotations

from typing import Any

from carm_roofline.benchmark.generation.code_gen import ControlInstructions, TypedInstructions, instruction as inst
from carm_roofline.benchmark.generation.code_gen.register import CyclicRegisterSet, HelperRegisterSet, TypedRegisterSets
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation
from carm_roofline.isa.base import BaseISA

# Scalar integer div uses the UNSIGNED div instruction (idiv is signed: its quotient must fit # the signed result range,
# which the remainder-invariant benchmark cannot guarantee). div faults only on a zero divisor or an overflowing
# unsigned quotient. The setup zeroes the dividend high half (so the first dividend fits) and preloads every divisor
# nonzero; the divisor registers are excluded from the dividend registers, so the quotient always fits.
_INT_DIV_ZERO = {
    DataType.i8: "movb $0, %ah",
    DataType.i16: "xorw %dx, %dx",
    DataType.i32: "xorl %edx, %edx",
    DataType.i64: "xorl %edx, %edx",
}
_INT_DIV_MOV = {DataType.i8: "movb", DataType.i16: "movw", DataType.i32: "movl", DataType.i64: "movq"}
_INT_DIV_IMM = {DataType.i8: 16, DataType.i16: 4096, DataType.i32: 4096, DataType.i64: 4096}
# div writes the quotient to the low dividend half and the remainder to the high half; both
# halves are implicitly clobbered and are no longer bench registers.
_INT_DIV_CLOBBER = {
    DataType.i8: ["%ax"],
    DataType.i16: ["%ax", "%dx"],
    DataType.i32: ["%eax", "%edx"],
    DataType.i64: ["%rax", "%rdx"],
}


def _vector_register_sets(prefix: str, count: int) -> dict[DataType, CyclicRegisterSet]:
    """Map every DataType to a fresh CyclicRegisterSet over `count` `%<prefix>{i}` registers."""
    return {dt: CyclicRegisterSet(f"%{prefix}{{}}", [(0, count - 1)]) for dt in DataType}


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
    # outer_iterator must not be RAX: scalar integer idiv implicitly writes RAX (quotient),
    # which would destroy a RAX-resident loop counter. r12 is caller-saved and safe.
    helper_registers = HelperRegisterSet("%{}", ("r12", "rdi", "r8", "r9", "r10"))

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class X86Scalar(BaseX86, register=True):
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
            DataType.i8: {
                ArithmeticOperation.add: "addb {}, {}",
                ArithmeticOperation.mul: "imulb {}",
                ArithmeticOperation.div: "divb {}",
                MemoryOperation.ld: "movb {off}({ptr}), {reg}",
                MemoryOperation.st: "movb {reg}, {off}({ptr})",
            },
            DataType.i16: {
                ArithmeticOperation.add: "addw {}, {}",
                ArithmeticOperation.mul: "imulw {}, {}",
                ArithmeticOperation.div: "divw {}",
                MemoryOperation.ld: "movw {off}({ptr}), {reg}",
                MemoryOperation.st: "movw {reg}, {off}({ptr})",
            },
            DataType.i32: {
                ArithmeticOperation.add: "addl {}, {}",
                ArithmeticOperation.mul: "imull {}, {}",
                ArithmeticOperation.div: "divl {}",
                MemoryOperation.ld: "movl {off}({ptr}), {reg}",
                MemoryOperation.st: "movl {reg}, {off}({ptr})",
            },
            DataType.i64: {
                ArithmeticOperation.add: "addq {}, {}",
                ArithmeticOperation.mul: "imulq {}, {}",
                ArithmeticOperation.div: "divq {}",
                MemoryOperation.ld: "movq {off}({ptr}), {reg}",
                MemoryOperation.st: "movq {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(
            {
                **_vector_register_sets("xmm", 16),
                # Why the integer GPR sets replace the vector defaults (div #DE trap-freedom,
                # REX high-byte limits, iterator/pointer regs): see carm_roofline/isa/AGENTS.md.
                DataType.i8: CyclicRegisterSet("%{}", ["bl", "cl", "dl", "sil", "bpl"]),
                DataType.i16: CyclicRegisterSet("%{}", ["bx", "cx", "si", "bp"]),
                DataType.i32: CyclicRegisterSet("%{}", ["ebx", "ecx", "esi", "ebp"]),
                DataType.i64: CyclicRegisterSet("%{}", ["rbx", "rcx", "rsi", "rbp", "r11", "r13", "r14", "r15"]),
            }
        )

    def setup_assembly(self, data_type: DataType) -> list[str]:
        """Preload GPRs so scalar integer idiv cannot fault (#DE): zero the dividend's high
        half and load every bench register (incl. the dividend low half) with a small
        nonzero value. Runs once per benchmark call, outside the timed loops."""
        if data_type not in _INT_DIV_MOV:
            return []
        lines = [_INT_DIV_ZERO[data_type]]
        for reg in self.bench_registers[data_type]:
            lines.append(f"{_INT_DIV_MOV[data_type]} ${_INT_DIV_IMM[data_type]}, {reg}")
        # These raw strings bypass the instruction .fmt() escaping; '%' must be doubled.
        return [inst.escape_for_inline_asm(line) for line in lines]

    def implicit_clobbers(self, data_type: DataType) -> list[str]:
        return _INT_DIV_CLOBBER.get(data_type, [])


class X86SSE(X86Scalar, register=True):
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
            DataType.i8: {
                ArithmeticOperation.add: "paddb {}, {}",
                MemoryOperation.ld: "movaps {off}({ptr}), {reg}",
                MemoryOperation.st: "movaps {reg}, {off}({ptr})",
            },
            DataType.i16: {
                ArithmeticOperation.add: "paddw {}, {}",
                ArithmeticOperation.mul: "pmullw {}, {}",
                MemoryOperation.ld: "movaps {off}({ptr}), {reg}",
                MemoryOperation.st: "movaps {reg}, {off}({ptr})",
            },
            DataType.i32: {
                ArithmeticOperation.add: "paddd {}, {}",
                ArithmeticOperation.mul: "pmulld {}, {}",
                MemoryOperation.ld: "movaps {off}({ptr}), {reg}",
                MemoryOperation.st: "movaps {reg}, {off}({ptr})",
            },
            DataType.i64: {
                ArithmeticOperation.add: "paddq {}, {}",
                MemoryOperation.ld: "movaps {off}({ptr}), {reg}",
                MemoryOperation.st: "movaps {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(_vector_register_sets("xmm", 16))

    def setup_assembly(self, data_type: DataType) -> list[str]:
        "SSE has no scalar integer idiv; the X86Scalar GPR preload must not run"
        return []

    def implicit_clobbers(self, data_type: DataType) -> list[str]:
        return []

    def bytes_per_inst(self, data_type: DataType) -> int:
        return self.BITS // 8

    def ops_per_inst(self, data_type: DataType, op: ArithmeticOperation) -> int:
        return op.ops() * self.BITS // data_type.bits()


class X86AVX(X86SSE, register=True):
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

        self.bench_registers = TypedRegisterSets(_vector_register_sets("ymm", 16))


class X86AVX2(X86AVX, register=True):
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
            DataType.i8: {
                ArithmeticOperation.add: "vpaddb {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i16: {
                ArithmeticOperation.add: "vpaddw {}, {}, {}",
                ArithmeticOperation.mul: "vpmullw {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i32: {
                ArithmeticOperation.add: "vpaddd {}, {}, {}",
                ArithmeticOperation.mul: "vpmulld {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i64: {
                ArithmeticOperation.add: "vpaddq {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class X86AVX512(X86SSE, register=True):
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
            DataType.i8: {
                ArithmeticOperation.add: "vpaddb {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i16: {
                ArithmeticOperation.add: "vpaddw {}, {}, {}",
                ArithmeticOperation.mul: "vpmullw {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i32: {
                ArithmeticOperation.add: "vpaddd {}, {}, {}",
                ArithmeticOperation.mul: "vpmulld {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.i64: {
                ArithmeticOperation.add: "vpaddq {}, {}, {}",
                ArithmeticOperation.mul: "vpmullq {}, {}, {}",  # AVX-512DQ (universal on mainstream AVX-512 CPUs)
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
            DataType.bf16: {
                ArithmeticOperation.fma: "vdpbf16ps {}, {}, {}",
                MemoryOperation.ld: "vmovaps {off}({ptr}), {reg}",
                MemoryOperation.st: "vmovaps {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(_vector_register_sets("zmm", 32))
