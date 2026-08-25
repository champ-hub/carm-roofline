from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from carm_roofline.architecture import arm as arm_architecture
from carm_roofline.architecture.detect import DetectedArchitecture
from carm_roofline.core import ArithmeticOperation, DataType, MemoryOperation
from carm_roofline.isa import ArmNeon, ArmSVE
from carm_roofline.isa.arm import ArmLoadImm


pytestmark = pytest.mark.unit


def test_neon_operations_scale_with_vector_width() -> None:
    neon = ArmNeon()

    assert neon.ops_per_inst(DataType.f32, ArithmeticOperation.add) == 4
    assert neon.ops_per_inst(DataType.f64, ArithmeticOperation.fma) == 4
    assert neon.bench_instructions.get(DataType.f32, ArithmeticOperation.add).fmt(
        neon.bench_registers[DataType.f32]
    ) == "fadd v0.4s, v0.4s, v0.4s"
    assert neon.bench_instructions.get(DataType.f64, MemoryOperation.ld).pattern == "ldr {reg}, [{ptr}, #{off}]"


def test_arm_load_immediate_emits_required_16_bit_words() -> None:
    loader = ArmLoadImm()

    assert loader.fmt("x3", 0x123400000001) == "movz x3, #1\\n\\tmovk x3, #4660, lsl #32"
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        loader.fmt("x3", -1)


def test_arm_detection_probes_vector_length_only_for_sve(monkeypatch: pytest.MonkeyPatch) -> None:
    generic_tests = Mock(return_value=DetectedArchitecture(isa=["arm_neon", "arm_sve"]))
    vector_probe = Mock(return_value={"vector_length": 16})
    monkeypatch.setattr(arm_architecture, "run_generic_tests", generic_tests)
    monkeypatch.setattr(arm_architecture, "detect_vlen", vector_probe)

    detected = arm_architecture.detect()

    assert detected.isa == ["arm", "arm_neon", "arm_sve"]
    assert detected.vector_length == 16
    assert generic_tests.call_args.kwargs["include_vlen"] is False
    vector_probe.assert_called_once()


def test_arm_detection_skips_vector_probe_without_sve(monkeypatch: pytest.MonkeyPatch) -> None:
    generic_tests = Mock(return_value=DetectedArchitecture(isa=["arm_neon"]))
    vector_probe = Mock(return_value={"vector_length": 16})
    monkeypatch.setattr(arm_architecture, "run_generic_tests", generic_tests)
    monkeypatch.setattr(arm_architecture, "detect_vlen", vector_probe)

    detected = arm_architecture.detect()

    assert detected.isa == ["arm", "arm_neon"]
    assert detected.vector_length is None
    vector_probe.assert_not_called()


def test_sve_uses_detected_vector_length() -> None:
    sve = ArmSVE.from_architecture(SimpleNamespace(vector_length=16))

    assert sve.vlen_bits == 128
    assert sve.ops_per_inst(DataType.f32, ArithmeticOperation.add) == 4
    assert sve.ops_per_inst(DataType.f64, ArithmeticOperation.fma) == 4


def test_sve_uses_predicated_arithmetic_and_typed_memory_instructions() -> None:
    sve = ArmSVE(vlen_bits=128)

    assert sve.bench_instructions.get(DataType.f32, ArithmeticOperation.fma).pattern == "fmla {}.s, p0/m, {}.s, {}.s"
    assert sve.bench_instructions.get(DataType.f64, MemoryOperation.ld).pattern == "ld1d {reg}.d, p0/z, [{ptr}, #{off}, mul vl]"
