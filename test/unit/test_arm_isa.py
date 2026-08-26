from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from carm_roofline.architecture import arm as arm_architecture
from carm_roofline.architecture.detect import DetectedArchitecture
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation
from carm_roofline.isa import ArmNeon, ArmScalar, ArmSVE
from carm_roofline.isa.arm import ArmLoadImm

pytestmark = pytest.mark.unit


def test_neon_operations_scale_with_vector_width() -> None:
    neon = ArmNeon()

    assert neon.ops_per_inst(DataType.f32, ArithmeticOperation.add) == 4
    assert neon.ops_per_inst(DataType.f64, ArithmeticOperation.fma) == 4
    assert (
        neon.bench_instructions.get(DataType.f32, ArithmeticOperation.add).fmt(neon.bench_registers[DataType.f32])
        == "fadd v0.4s, v0.4s, v0.4s"
    )
    assert neon.bench_instructions.get(DataType.f64, MemoryOperation.ld).pattern == "ldr {reg}, [{ptr}, #{off}]"


def test_scalar_integer_instructions_use_native_register_widths() -> None:
    scalar = ArmScalar()

    assert (
        scalar.bench_instructions.get(DataType.i8, ArithmeticOperation.fma).fmt(scalar.bench_registers[DataType.i8])
        == "madd w5, w5, w5, w5"
    )
    assert scalar.bench_instructions.get(DataType.i16, MemoryOperation.ld).pattern == "ldrh {reg}, [{ptr}, #{off}]"
    assert scalar.bench_instructions.get(DataType.i32, MemoryOperation.ld).pattern == "ldr {reg}, [{ptr}, #{off}]"
    assert (
        scalar.bench_instructions.get(DataType.i64, ArithmeticOperation.div).fmt(scalar.bench_registers[DataType.i64])
        == "sdiv x5, x5, x5"
    )
    for data_type in (DataType.i8, DataType.i16, DataType.i32, DataType.i64):
        assert all(register not in {"w0", "w1", "w2", "w3", "w4", "w30", "w31", "x0", "x1", "x2", "x3", "x4", "x30", "x31"} for register in scalar.bench_registers[data_type])


@pytest.mark.parametrize(
    "data_type,suffix,element_count,has_multiply_add",
    [
        (DataType.i8, "16b", 16, True),
        (DataType.i16, "8h", 8, True),
        (DataType.i32, "4s", 4, True),
        (DataType.i64, "2d", 2, False),
    ],
)
def test_neon_integer_lanes_and_instructions(
    data_type: DataType, suffix: str, element_count: int, has_multiply_add: bool
) -> None:
    neon = ArmNeon()

    assert neon.ops_per_inst(data_type, ArithmeticOperation.add) == element_count
    assert (
        neon.bench_instructions.get(data_type, ArithmeticOperation.add).fmt(neon.bench_registers[data_type])
        == f"add v4.{suffix}, v4.{suffix}, v4.{suffix}"
    )
    assert neon.bench_instructions.get(data_type, MemoryOperation.ld).pattern == "ldr {reg}, [{ptr}, #{off}]"
    if has_multiply_add:
        fma_neon = ArmNeon()
        assert neon.ops_per_inst(data_type, ArithmeticOperation.fma) == element_count * 2
        assert (
            neon.bench_instructions.get(data_type, ArithmeticOperation.fma).fmt(fma_neon.bench_registers[data_type])
            == f"mla v4.{suffix}, v4.{suffix}, v4.{suffix}"
        )


@pytest.mark.parametrize(
    "data_type,arith_suffix,memory_suffix,element_count,has_div",
    [
        (DataType.i8, "b", "b", 16, False),
        (DataType.i16, "h", "h", 8, False),
        (DataType.i32, "s", "w", 4, True),
        (DataType.i64, "d", "d", 2, True),
    ],
)
def test_sve_integer_lanes_and_predicated_instructions(
    data_type: DataType, arith_suffix: str, memory_suffix: str, element_count: int, has_div: bool
) -> None:
    sve = ArmSVE(vlen_bits=128)

    assert sve.ops_per_inst(data_type, ArithmeticOperation.add) == element_count
    assert sve.ops_per_inst(data_type, ArithmeticOperation.fma) == element_count * 2
    assert sve.bench_instructions.get(data_type, ArithmeticOperation.fma).pattern == (
        f"mla {{}}.{arith_suffix}, p0/m, {{}}.{arith_suffix}, {{}}.{arith_suffix}"
    )
    assert sve.bench_instructions.get(data_type, MemoryOperation.ld).pattern == (
        f"ld1{memory_suffix} {{reg}}.{arith_suffix}, p0/z, [{{ptr}}, #{{off}}, mul vl]"
    )
    if has_div:
        assert sve.ops_per_inst(data_type, ArithmeticOperation.div) == element_count


@pytest.mark.parametrize(
    "data_type,predicate",
    [
        (DataType.i8, "b"),
        (DataType.i16, "h"),
        (DataType.i32, "s"),
        (DataType.i64, "d"),
        (DataType.f32, "s"),
        (DataType.f64, "d"),
    ],
)
def test_sve_predicate_setup_matches_element_width(data_type: DataType, predicate: str) -> None:
    assert ArmSVE(vlen_bits=128).setup_assembly(data_type) == [f"ptrue p0.{predicate}"]


def test_arm_load_immediate_emits_required_16_bit_words() -> None:
    loader = ArmLoadImm()

    assert loader.fmt("x3", 0x123400000001) == ["movz x3, #1", "movk x3, #4660, lsl #32"]
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        loader.fmt("x3", -1)


def test_arm_detection_probes_vector_length_only_for_sve(monkeypatch: pytest.MonkeyPatch) -> None:
    generic_tests = Mock(return_value=DetectedArchitecture(isa=["arm_neon", "arm_sve"], vector_length=16))
    monkeypatch.setattr(arm_architecture, "run_generic_tests", generic_tests)

    detected = arm_architecture.detect()

    assert detected.isa == ["arm", "arm_neon", "arm_sve"]
    assert detected.vector_length == 16


def test_arm_detection_skips_vector_probe_without_sve(monkeypatch: pytest.MonkeyPatch) -> None:
    generic_tests = Mock(return_value=DetectedArchitecture(isa=["arm_neon"]))
    monkeypatch.setattr(arm_architecture, "run_generic_tests", generic_tests)

    detected = arm_architecture.detect()

    assert detected.isa == ["arm", "arm_neon"]
    assert detected.vector_length is None


def test_sve_uses_detected_vector_length() -> None:
    sve = ArmSVE.from_architecture(SimpleNamespace(vector_length=16))

    assert sve.vlen_bits == 128
    assert sve.ops_per_inst(DataType.f32, ArithmeticOperation.add) == 4
    assert sve.ops_per_inst(DataType.f64, ArithmeticOperation.fma) == 4


def test_sve_uses_predicated_arithmetic_and_typed_memory_instructions() -> None:
    sve = ArmSVE(vlen_bits=128)

    assert sve.bench_instructions.get(DataType.f32, ArithmeticOperation.fma).pattern == "fmla {}.s, p0/m, {}.s, {}.s"
    assert (
        sve.bench_instructions.get(DataType.f64, MemoryOperation.ld).pattern
        == "ld1d {reg}.d, p0/z, [{ptr}, #{off}, mul vl]"
    )
