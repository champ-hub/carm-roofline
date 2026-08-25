from __future__ import annotations

from typing import Any

from carm_roofline.benchmark.generation.code_gen import ControlInstructions, TypedInstructions, instruction as inst
from carm_roofline.benchmark.generation.code_gen.instruction import _Instruction
from carm_roofline.benchmark.generation.code_gen.register import CyclicRegisterSet, HelperRegisterSet, TypedRegisterSets
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation, Operation
from carm_roofline.isa.base import BaseISA


class BaseArm(BaseISA):
    name = "arm"
    family = "arm"
    max_branch_insts = 2**23 // 4  # signed 24-bit immediate byte offset, 4-byte insts
    max_mem_offset_bytes = 2**12  # unsigned 12-bit immediate (sign in another field perhaps?)
    max_immediate = 2**12 - 1  # unsigned 12-bit

    class ControlInsts(ControlInstructions):
        load_imm = inst.LoadImm("mov {dst}, {imm}")
        load_word = inst.LoadWord("ldr {dst}, {ptr}")
        branch_nz = inst.BranchNotZero("cbnz {src}, {tgt}")
        add_imm = inst.AddImm("add {src}, {src}, {imm}")
        sub_imm = inst.SubImm("sub {src}, {src}, {imm}")
        add = inst.Add("add {dst}, {src1}, {src2}")

    control_instructions = ControlInsts()
    helper_registers = HelperRegisterSet("x{}", (0, 1, 2, 3, 4))

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


def _make_float_instructions_scalar() -> dict[Operation, str | _Instruction]:
    return {
        ArithmeticOperation.add: "fadd {}, {}, {}",
        ArithmeticOperation.mul: "fmul {}, {}, {}",
        ArithmeticOperation.div: "fdiv {}, {}, {}",
        ArithmeticOperation.fma: "fmadd {}, {}, {}, {}",
        MemoryOperation.ld: "ldr {reg}, [{ptr}, #{off}]",
        MemoryOperation.st: "str {reg}, [{ptr}, #{off}]",
    }


class ArmScalar(BaseArm, register=True):
    name = "arm"

    bench_instructions = TypedInstructions(
        {
            DataType.f32: _make_float_instructions_scalar(),
            DataType.f64: _make_float_instructions_scalar(),
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: CyclicRegisterSet("s{}", [(0, 31)]),
                DataType.f64: CyclicRegisterSet("d{}", [(0, 31)]),
                DataType.i32: CyclicRegisterSet("w{}", [(4, 31)]),
                DataType.i64: CyclicRegisterSet("x{}", [(4, 31)]),
            }
        )


def _make_float_instructions_neon(suf: str) -> dict[Operation, str | _Instruction]:
    return {
        ArithmeticOperation.add: inst.Arithmetic(f"fadd {{}}.{suf}, {{}}.{suf}, {{}}.{suf}", register_format="v{}"),
        ArithmeticOperation.mul: inst.Arithmetic(f"fmul {{}}.{suf}, {{}}.{suf}, {{}}.{suf}", register_format="v{}"),
        ArithmeticOperation.div: inst.Arithmetic(f"fdiv {{}}.{suf}, {{}}.{suf}, {{}}.{suf}", register_format="v{}"),
        ArithmeticOperation.fma: inst.Arithmetic(f"fmla {{}}.{suf}, {{}}.{suf}, {{}}.{suf}", register_format="v{}"),
        MemoryOperation.ld: "ldr {reg}, [{ptr}, #{off}]",
        MemoryOperation.st: "str {reg}, [{ptr}, #{off}]",
    }


class ArmNeon(BaseArm, register=True):
    name = "arm_neon"

    bench_instructions = TypedInstructions(
        {
            DataType.f32: _make_float_instructions_neon("4s"),
            DataType.f64: _make_float_instructions_neon("2d"),
        }
    )

    def __init__(self, **kwargs: Any):
        super().__init__()

        self.bench_registers = TypedRegisterSets(
            {
                DataType.f32: CyclicRegisterSet("q{}", [(0, 31)]),
                DataType.f64: CyclicRegisterSet("q{}", [(0, 31)]),
                DataType.i32: CyclicRegisterSet("q{}", [(4, 31)]),
                DataType.i64: CyclicRegisterSet("q{}", [(4, 31)]),
            }
        )

    def bytes_per_inst(self, data_type: DataType) -> int:
        return 128 // 8


def _make_float_instructions_sve(suf: str) -> dict[Operation, str | _Instruction]:
    return {
        ArithmeticOperation.add: f"fadd {{}}.{suf}, {{}}.{suf}, {{}}.{suf}",
        ArithmeticOperation.mul: f"fmul {{}}.{suf}, {{}}.{suf}, {{}}.{suf}",
        ArithmeticOperation.div: f"fdiv {{}}.{suf}, {{}}.{suf}, {{}}.{suf}",
        ArithmeticOperation.fma: f"fmla {{}}.{suf}, {{}}.{suf}, {{}}.{suf}",
        MemoryOperation.ld: f"ld1w {{reg}}.{suf}, p0/z, [{{ptr}}, #{{off}}, mul vl]",
        MemoryOperation.st: f"st1w {{reg}}.{suf}, p0, [{{ptr}}, #{{off}}, mul vl]",
    }


class ArmSVE(BaseArm, register=True):
    name = "arm_sve"

    bench_instructions = TypedInstructions(
        {
            DataType.f32: _make_float_instructions_sve("s"),
            DataType.f64: _make_float_instructions_sve("d"),
        }
    )

    def __init__(self, vlen_bits: int, **kwargs: Any):
        super().__init__(**kwargs)

        self.vlen_bits = vlen_bits

        # SVE's maximum offset is very small, so unrolling loops is necessary
        self.unroll_loop = True

        if vlen_bits < 1:
            raise ValueError("Vector length 'vlen_bits' must be a positive integer")

        # Create fresh CyclicRegisterSet instances per instance (mutable state)
        registers = CyclicRegisterSet("z{}", [(0, 31)])
        self.bench_registers = TypedRegisterSets(dict.fromkeys(DataType, registers))

    # SVE's offsets are not scaled by the data type size
    def offset_increment(self, data_type: DataType) -> int:
        return 1

    # The offsets are vlen-relative, with a max value of 7
    def max_unique_offsets(self, data_type: DataType) -> int:
        return 8

    # Setup the predicate register with all 1's
    def setup_assembly(self, data_type: DataType) -> list[str]:
        type_str = "s" if data_type == DataType.f32 else "d"
        return [f"ptrue p0.{type_str}"]

    def bytes_per_inst(self, data_type: DataType) -> int:
        # SVE vector width in bytes (independent of data type)
        return self.vlen_bits // 8
