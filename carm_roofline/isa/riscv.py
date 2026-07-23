from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from carm_roofline.benchmark.generation.code_gen import ControlInstructions, TypedInstructions, instruction as inst
from carm_roofline.benchmark.generation.code_gen.instruction import _Instruction
from carm_roofline.benchmark.generation.code_gen.register import CyclicRegisterSet, HelperRegisterSet, TypedRegisterSets
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation, Operation
from carm_roofline.isa.base import BaseISA, InlineASM

if TYPE_CHECKING:
    from carm_roofline.architecture.architecture import Architecture


class BaseRISCV(BaseISA):
    # Define the control instructions
    class ControlInsts(ControlInstructions):
        load_imm = inst.LoadImm("li {dst}, {imm}")
        load_word = inst.LoadWord("ld {dst}, {ptr}")
        # Will work with bgtz or bnz, as documented in the class
        branch_nz = inst.BranchNotZero("bgtz {src}, {tgt}")
        add_imm = inst.AddImm("addi {dst}, {src}, {imm}")
        sub_imm = inst.SubImm("addi {dst}, {src}, -{imm}")
        add = inst.Add("add {dst}, {src1}, {src2}")

    name = "riscv"
    family = "riscv"
    # Required instruction formats
    control_instructions = ControlInsts()
    # Registers for iterator, pointer, etc. operations
    helper_registers = HelperRegisterSet("x{}", (5, 6, 7, 9, 28))
    # Maximum branch distance in instructions
    # signed 12-bit offset in multiples of two bytes, i.e. +-4 KiB
    # divided by 4 bytes per instruction results in 1024 instructions
    max_branch_insts = 1024
    # Maximum memory offset in bytes
    # signed 12-bit offset, i.e. a +-2^11 = 2 KiB range
    max_mem_offset_bytes = 2**11
    # Maximum immediate for computational instructions (i.g. addi)
    max_immediate = 2**11

    # RISC-V-specific inline asm input formatting (needs to be %(var) and not %[var])
    def format_iasm_input(self, var: InlineASM.Input) -> str:
        return f"%({var.asm_name})"


class RISCVScalar(BaseRISCV, register=True):
    # Instructions associated with each data type and operation
    # Instructions are immutable, so we can define them directly here
    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                ArithmeticOperation.add: "fadd.s {}, {}, {}",
                ArithmeticOperation.mul: "fmul.s {}, {}, {}",
                ArithmeticOperation.div: "fdiv.s {}, {}, {}",
                ArithmeticOperation.fma: "fmadd.s {}, {}, {}, {}",
                MemoryOperation.ld: "flw {reg}, {off}({ptr})",
                MemoryOperation.st: "fsw {reg}, {off}({ptr})",
            },
            DataType.f64: {
                ArithmeticOperation.add: "fadd.d {}, {}, {}",
                ArithmeticOperation.mul: "fmul.d {}, {}, {}",
                ArithmeticOperation.div: "fdiv.d {}, {}, {}",
                ArithmeticOperation.fma: "fmadd.d {}, {}, {}, {}",
                MemoryOperation.ld: "fld {reg}, {off}({ptr})",
                MemoryOperation.st: "fsd {reg}, {off}({ptr})",
            },
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Benchmark registers
        # Registers are mutable (internal counter), so they must be created per-instance
        registers_float = CyclicRegisterSet("f{}", [(0, 31)])
        registers_int = CyclicRegisterSet("x{}", [(10, 31)])

        # Registers associated with each data type
        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: registers_float,
                DataType.f64: registers_float,
                DataType.i32: registers_int,
                DataType.i64: registers_int,
            }
        )


def _make_float_instructions_rvv() -> dict[Operation, str | _Instruction]:
    "Helper to generate the floating-point instructions, which are identical for f32 and f64."
    return {
        ArithmeticOperation.add: "vfadd.vv {}, {}, {}",
        ArithmeticOperation.mul: "vfmul.vv {}, {}, {}",
        ArithmeticOperation.div: "vfdiv.vv {}, {}, {}",
        ArithmeticOperation.fma: "vfmadd.vv {}, {}, {}, {}",
        MemoryOperation.ld: "vle.v {reg}, ({ptr})",
        MemoryOperation.st: "vse.v {reg}, ({ptr})",
    }


class RISCV_RVV_071(RISCVScalar, register=True):
    name = "riscv_rvv_0_7_1"

    TYPE_TO_VSETVL: Mapping[DataType, str] = {
        DataType.f32: "e32",
        DataType.i32: "e32",
        DataType.f64: "e64",
        DataType.i64: "e64",
    }

    @classmethod
    def from_architecture(cls, arch: Architecture) -> RISCV_RVV_071:
        """Create RVV instance with vector length and LMUL from architecture.

        Args:
            arch: Architecture instance with detected vector_length and vector_lmul.

        Returns:
            RISCV_RVV_071 instance configured with target vector parameters.

        Raises:
            ValueError: If vector_length or vector_lmul not available in architecture.
        """
        if arch.vector_length is None:
            raise ValueError("Vector length not detected/specified. Provide --vector-length for RVV support.")
        lmul = arch.vector_lmul or 1
        return cls(vlen_bits=arch.vector_length, lmul=lmul)

    # Instructions associated with each data type and operation
    bench_instructions = TypedInstructions(
        {
            DataType.f32: _make_float_instructions_rvv(),
            DataType.f64: _make_float_instructions_rvv(),
        }
    )

    def __init__(self, vlen_bits: int, lmul: int, **kwargs: Any):
        super().__init__(**kwargs)

        self.vlen_bits = vlen_bits
        self.lmul = lmul

        self.unroll_loop = True

        if vlen_bits < 1:
            raise ValueError("Vector length 'vlen_bits' must be a positive integer")
        if lmul & (lmul - 1) != 0 or lmul == 0:
            raise ValueError("LMUL must be a power of two and non-zero")

        # generate the registers with regard to the lmul
        indices = list(range(32))[::lmul]
        registers = CyclicRegisterSet("v{}", indices)

        self.bench_registers = TypedRegisterSets(dict.fromkeys(DataType, registers))

    def setup_assembly(self, data_type: DataType) -> list[str]:
        elem_size = self.TYPE_TO_VSETVL[data_type]
        # We can reuse the outer iter here, not overwriting anything
        return [
            self.control_instructions.load_imm.fmt(self.helper_registers.outer_iterator, self.vlen_bits),
            f"vsetvli x0, {self.helper_registers.outer_iterator}, {elem_size}, m{self.lmul}",
        ]

    def bytes_per_inst(self, data_type: DataType) -> int:
        # Calculate bytes per instruction: vector width in bytes times lmul
        return (self.vlen_bits // 8) * self.lmul

    def ops_per_inst(self, data_type: DataType, op: ArithmeticOperation) -> int:
        elements = self.vlen_bits // data_type.bits() * self.lmul
        return op.ops() * elements


class RISCV_RVV(RISCV_RVV_071, register=True):
    name = "riscv_rvv"

    # Instructions associated with each data type and operation
    bench_instructions = TypedInstructions(
        {
            DataType.f32: {
                **RISCV_RVV_071.bench_instructions[DataType.f32],
                MemoryOperation.ld: "vle32.v {reg}, ({ptr})",  # override RVV 0.7.1 loads and stores
                MemoryOperation.st: "vse32.v {reg}, ({ptr})",
            },
            DataType.f64: {
                **RISCV_RVV_071.bench_instructions[DataType.f64],
                MemoryOperation.ld: "vle64.v {reg}, ({ptr})",
                MemoryOperation.st: "vse64.v {reg}, ({ptr})",
            },
        }
    )

    @classmethod
    def from_architecture(cls, arch: Architecture) -> RISCV_RVV:
        """Create RVV instance with vector length and LMUL from architecture.

        Args:
            arch: Architecture instance with detected vector_length and vector_lmul.

        Returns:
            RISCV_RVV instance configured with target vector parameters.

        Raises:
            ValueError: If vector_length or vector_lmul not available in architecture.
        """
        if arch.vector_length is None:
            raise ValueError("Vector length not detected/specified. Provide --vector-length for RVV support.")
        lmul = arch.vector_lmul or 1
        return cls(vlen_bits=arch.vector_length, lmul=lmul)

    def __init__(self, vlen_bits: int, lmul: int, **kwargs: Any):
        super().__init__(vlen_bits, lmul, **kwargs)
